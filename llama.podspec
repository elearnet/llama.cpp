Pod::Spec.new do |s|
  s.name             = 'llama'
  s.version          = '0.0.1'
  s.summary          = 'llama.cpp for ELUI'
  s.description      = 'Inference of Llama model in pure C/C++'
  s.homepage         = 'https://github.com/ggerganov/llama.cpp'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Georgi Gerganov' => 'https://github.com/ggerganov' }
  s.source           = { :path => '.' }

  s.ios.deployment_target = '13.0'
  s.osx.deployment_target = '11.0'

  # llama.cpp uses manual memory management (retain/release) in its Metal implementation.
  # so we disable ARC to avoid compiler errors with manual memory calls.
  s.requires_arc = false

  s.source_files = [
    'src/*.{h,cpp}',                           # Core llama.cpp implementation and public API
    'src/models/*.{h,cpp}',                    # Model-specific architectures (T5, Gemma, etc.)
    'ggml/src/*.{h,c,cpp}',                    # GGML base library (tensors, graphs, backends)
    'ggml/src/ggml-cpu/*.{h,c,cpp}',           # CPU computation backend (mandatory)
    'ggml/src/ggml-cpu/arch/*.{h,c,cpp}',      # Architecture-specific CPU optimizations (SIMD)
    'ggml/src/ggml-cpu/arch/arm/*.{h,c,cpp}',  # ARM-specific kernels (NEON)
    'ggml/src/ggml-cpu/arch/x86/*.{h,c,cpp}',  # x86-specific kernels (AVX)
    'ggml/src/ggml-cpu/llamafile/*.{h,c,cpp}', # Llamafile integration and CPU backend registration
    'ggml/src/ggml-metal/*.{h,m,mm,cpp}',      # Metal (Apple GPU) acceleration implementation
    'tools/mtmd/mtmd.{h,cpp}',                 # Multimodal context (mtmd_context), chunking, and tokenization
    'tools/mtmd/mtmd-audio.{h,cpp}',           # Multimodal audio encoding and audio chunk handling
    'tools/mtmd/mtmd-image.{h,cpp}',           # Image preprocessing, bitmap loading, and patch extraction
    'tools/mtmd/mtmd-helper.{h,cpp}',          # High-level helpers for batch evaluation into KV cache
    'tools/mtmd/clip.{h,cpp}',                 # Core CLIP vision encoder API and context management
    'tools/mtmd/clip-impl.h',                  # Internal CLIP tensor operations and helper macros
    'tools/mtmd/clip-model.h',                 # CLIP architecture metadata, hyperparameters, and tensor mapping
    'tools/mtmd/clip-graph.h',                 # Base class for building vision encoder computation graphs
    'tools/mtmd/models/*.{h,cpp}'              # Vision model backends (SigLIP, Qwen2VL, Pixtral, Gemma4V, etc.)
  ]

  s.resources = [
    'ggml/src/ggml-metal/*.metal',
    'ggml/src/ggml-common.h',
    'ggml/src/ggml-metal/ggml-metal-impl.h'
  ]

  s.public_header_files = [
    'include/*.h',
    'ggml/include/*.h',
    'tools/mtmd/mtmd.h',
    'tools/mtmd/mtmd-helper.h',
    'tools/mtmd/clip.h'
  ]

  s.header_mappings_dir = '.'

  # Dynamically fetch version and commit from Git
  git_version = `git -C "#{s.source[:path]}" describe --tags 2>/dev/null || git -C "#{s.source[:path]}" rev-parse --short HEAD 2>/dev/null || echo "unknown"`.strip
  git_commit = `git -C "#{s.source[:path]}" rev-parse --short HEAD 2>/dev/null || echo "unknown"`.strip

  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/include"',
      '"$(PODS_TARGET_SRCROOT)/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/src"',
      '"$(PODS_TARGET_SRCROOT)/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/tools/mtmd"',
      '"$(PODS_TARGET_SRCROOT)/vendor"'
    ].join(' '),
    'GCC_PREPROCESSOR_DEFINITIONS' => [
      'GGML_USE_CPU=1',         # Enables the CPU backend (required for all models)
      'GGML_USE_ACCELERATE=1',  # Uses Apple's Accelerate framework for optimized CPU math
      'GGML_USE_METAL=1',       # Enables Metal (GPU) acceleration on Apple platforms
      'GGML_METAL_USE_BF16=1',  # Enables BFloat16 support in Metal kernels (required for Gemma 4)
      'GGML_METAL_NDEBUG=1',    # Disables debug logging for Metal kernels (improves performance)
      # GGML_VERSION and GGML_COMMIT are required by ggml.c for version reporting.
      "GGML_VERSION=\\\"#{git_version}\\\"",
      "GGML_COMMIT=\\\"#{git_commit}\\\""
    ].join(' '),
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    # what are these two for? docs/DevDocs/devLogs/2026-06-25.md
    'OTHER_CFLAGS' => '$(inherited) -O3 -DNDEBUG',
    'OTHER_CPLUSPLUSFLAGS' => '$(inherited) -O3 -DNDEBUG',
    'OTHER_LDFLAGS' => '-framework Accelerate -framework Metal -framework MetalKit -framework Foundation'
  }

  s.weak_frameworks = 'Accelerate', 'Metal', 'MetalKit', 'Foundation'
end
