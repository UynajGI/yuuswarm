# Justfile for swarmlators project
default:
    @just --list

# Build and install Rust extension in dev mode
build-rust dev:
    @echo "📦 Building rust_engine in development mode..."
    cd rust_engine && maturin develop

rebuild:
    just clean
    just build-rust dev

clean:
    @echo "🧹 Cleaning Rust build artifacts..."
    cd rust_engine && cargo clean

install-maturin:
    @echo "⚙️ Installing maturin..."
    uv add maturin

test-rust:
    @echo "🧪 Testing rust_engine import..."
    python -c "import rust_engine; print('✅ rust_engine imported successfully!')"

setup:
    just install-maturin
    just build-rust
    just test-rust