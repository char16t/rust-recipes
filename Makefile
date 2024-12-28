build:
	cargo build

test:
	cargo test

lint:
	cargo clippy

check-format:
	cargo fmt --check

codecov: clean
	CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' LLVM_PROFILE_FILE='cargo-test-%p-%m.profraw' cargo test
	grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o target/coverage/html
	firefox --new-tab --url ./target/coverage/html/index.html

clean:
	rm -rf target/
	rm -rf *.profraw
	rm -rf *.input
