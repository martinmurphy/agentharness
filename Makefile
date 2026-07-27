IMAGE ?= agentharness
SKILLS ?= $(CURDIR)/skills
WORKSPACE ?= $(CURDIR)/workspace

# The workspace is mounted read-write, so the container user (UID 1001) has to
# be able to write a host-owned directory. Under rootless podman UID 1001 maps
# to a subuid by default, which cannot; this maps the invoking user onto it
# instead. Override with USERNS= (empty) if your setup doesn't need it.
USERNS ?= --userns=keep-id:uid=1001,gid=0

.PHONY: help build run shell test lint list-skills

help:
	@echo "targets:"
	@echo "  build        build the container image ($(IMAGE))"
	@echo "  run          run the REPL in the container (mounts ./skills ro + ./workspace rw)"
	@echo "  list-skills  run --list-skills in the container (no API key needed)"
	@echo "  shell        open a shell in the image"
	@echo "  test         run the test suite (host, needs .venv or pytest on PATH)"
	@echo "  lint         run ruff (host)"

build:
	podman build -t $(IMAGE) .

run:
	mkdir -p "$(WORKSPACE)"
	podman run --rm -it $(USERNS) \
		-e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e GEMINI_API_KEY \
		-v "$(SKILLS):/skills:ro,Z" \
		-v "$(WORKSPACE):/workspace:Z" \
		$(IMAGE)

list-skills:
	podman run --rm \
		-v "$(SKILLS):/skills:ro,Z" \
		$(IMAGE) --list-skills

shell:
	podman run --rm -it --entrypoint /bin/bash $(IMAGE)

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check .
