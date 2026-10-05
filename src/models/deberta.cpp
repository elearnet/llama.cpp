//deberta-v2 support bg
#include "models.h"
#include <cmath>

llm_build_deberta::llm_build_deberta(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Token embeddings
    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    // LayerNorm on token embeddings
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);
    cb(inpL, "inp_norm", 0);

    auto * inp_attn = build_attn_inp_no_cache();

    // Shared relative position embeddings (512, hidden_size)
    const int64_t n_rel = hparams.n_rel_attn_bkts * 2; // 512
    GGML_ASSERT(model.rel_embd != nullptr);
    GGML_ASSERT(model.rel_embd->ne[1] == n_rel);

    // Apply LayerNorm to shared relative position embeddings
    ggml_tensor * rel_embd = model.rel_embd->type == GGML_TYPE_F32 ? model.rel_embd : ggml_cast(ctx0, model.rel_embd, GGML_TYPE_F32);
    ggml_tensor * P = build_norm(rel_embd, model.rel_embd_norm, model.rel_embd_norm_b, LLM_NORM, -1);
    cb(P, "rel_embd_norm", -1);

    // Relative position indices for C2P and P2C
    auto [c2p_idx, p2c_idx] = build_inp_deberta_rel_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        cur = inpL;

        // Content Q, K, V
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // Project relative position embeddings through key_proj (wk) and query_proj (wq)
        // P: [n_embd, n_rel]
        // wk: [n_embd, n_embd] -> Kr: [n_embd, n_rel]
        ggml_tensor * Kr = build_lora_mm(model.layers[il].wk, P);
        if (model.layers[il].wk_b) {
            Kr = ggml_add(ctx0, Kr, model.layers[il].wk_b);
        }
        Kr = ggml_reshape_3d(ctx0, Kr, n_embd_head, n_head, n_rel);
        cb(Kr, "Kr", il);

        ggml_tensor * Qr = build_lora_mm(model.layers[il].wq, P);
        if (model.layers[il].wq_b) {
            Qr = ggml_add(ctx0, Qr, model.layers[il].wq_b);
        }
        Qr = ggml_reshape_3d(ctx0, Qr, n_embd_head, n_head, n_rel);
        cb(Qr, "Qr", il);

        // Permute Kr and Qr so head dimension is on dimension 2 for batch mul_mat:
        // Kr: [n_embd_head, n_head, n_rel] -> [n_embd_head, n_rel, n_head]
        ggml_tensor * Kr_perm = ggml_permute(ctx0, Kr, 0, 2, 1, 3);
        ggml_tensor * Qr_perm = ggml_permute(ctx0, Qr, 0, 2, 1, 3);

        // Qcur: [n_embd_head, n_head, n_tokens] -> [n_embd_head, n_tokens, n_head]
        ggml_tensor * Q_perm = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        // Kcur: [n_embd_head, n_head_kv, n_tokens] -> [n_embd_head, n_tokens, n_head_kv]
        ggml_tensor * K_perm = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);

        // ─── C2P: Q_content @ K_pos^T ───
        // mul_mat(Kr_perm, Q_perm) -> [n_rel, n_tokens, n_head, 1]
        // ne[0] = n_rel, ne[1] = n_tokens, ne[2] = n_head
        ggml_tensor * c2p_raw = ggml_mul_mat(ctx0, Kr_perm, Q_perm);
        cb(c2p_raw, "c2p_raw", il);

        // Permute to [n_head, n_rel, n_tokens]:
        // ne[0] = n_head, ne[1] = n_rel, ne[2] = n_tokens
        ggml_tensor * c2p_perm = ggml_permute(ctx0, c2p_raw, 1, 2, 0, 3);
        ggml_tensor * c2p_cont = ggml_cont(ctx0, c2p_perm);
        // Reshape to 2D table: [n_head, n_tokens * n_rel]
        // In contiguous memory, row offset is i * n_rel + r
        ggml_tensor * c2p_table = ggml_reshape_2d(ctx0, c2p_cont, n_head, n_tokens * n_rel);

        // Gather using c2p_idx: selects columns from c2p_table
        // c2p_idx has shape [n_tokens * n_tokens] with flat index i * n_rel + bucket(i - j)
        ggml_tensor * A_c2p = ggml_get_rows(ctx0, c2p_table, c2p_idx);
        // A_c2p shape: [n_head, n_tokens * n_tokens]
        // Reshape to 3D: [n_head, n_tokens (j), n_tokens (i)]
        A_c2p = ggml_reshape_3d(ctx0, A_c2p, n_head, n_tokens, n_tokens);
        // Permute to match kq shape: [n_tokens (j), n_tokens (i), n_head, 1]
        // ne[0]=n_head goes to axis 2, ne[1]=n_tokens goes to axis 0, ne[2]=n_tokens goes to axis 1
        A_c2p = ggml_permute(ctx0, A_c2p, 2, 0, 1, 3);
        A_c2p = ggml_cont(ctx0, A_c2p);
        cb(A_c2p, "A_c2p", il);

        // ─── P2C: K_content @ Q_pos^T ───
        // mul_mat(Qr_perm, K_perm) -> [n_rel, n_tokens, n_head, 1]
        ggml_tensor * p2c_raw = ggml_mul_mat(ctx0, Qr_perm, K_perm);
        cb(p2c_raw, "p2c_raw", il);

        ggml_tensor * p2c_perm = ggml_permute(ctx0, p2c_raw, 1, 2, 0, 3);
        ggml_tensor * p2c_cont = ggml_cont(ctx0, p2c_perm);
        ggml_tensor * p2c_table = ggml_reshape_2d(ctx0, p2c_cont, n_head, n_tokens * n_rel);

        // Gather using p2c_idx: selects columns with flat index j * n_rel + bucket(j - i)
        ggml_tensor * A_p2c = ggml_get_rows(ctx0, p2c_table, p2c_idx);
        A_p2c = ggml_reshape_3d(ctx0, A_p2c, n_head, n_tokens, n_tokens);
        // Permute to match kq shape: [n_tokens (j), n_tokens (i), n_head, 1]
        A_p2c = ggml_permute(ctx0, A_p2c, 2, 0, 1, 3);
        A_p2c = ggml_cont(ctx0, A_p2c);
        cb(A_p2c, "A_p2c", il);

        // Disentangled attention bias: kq_b = A_c2p + A_p2c
        ggml_tensor * kq_b = ggml_add(ctx0, A_c2p, A_p2c);
        cb(kq_b, "kq_b", il);

        // Self-attention with scaling = 1 / sqrt(3 * n_embd_head)
        const float kq_scale = 1.0f / sqrtf(3.0f * (float)n_embd_head);

        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].wo_b, nullptr,
                Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // Residual + attention output LayerNorm
        cur = ggml_add(ctx0, cur, inpL);
        cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // Feed-Forward Network: GELU activation (intermediate -> GELU -> output)
        cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, nullptr,
                nullptr, nullptr, nullptr,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr, nullptr,
                LLM_FFN_GELU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        // Residual + layer output LayerNorm
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

        inpL = cur;
    }

    // Output token embeddings
    res->t_embd = inpL;
    cb(res->t_embd, "t_embd", -1);
    ggml_build_forward_expand(gf, res->t_embd);
}
//deberta-v2 support ed