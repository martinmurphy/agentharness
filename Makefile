IMAGE ?= agentharness
SKILLS ?= $(CURDIR)/skills

.PHONY: help build run shell test lint list-skills

help:
	@echo "targets:"
	@echo "  build        build the container image ($(IMAGE))"
	@echo "  run          run the REPL in the container (mounts ./skills, passes API keys)"
	@echo "  list-skills  run --list-skills in the container (no API key needed)"
	@echo "  shell        open a shell in the image"
	@echo "  test         run the test suite (host, needs .venv or pytest on PATH)"
	@echo "  lint         run ruff (host)"

build:
	podman build -t $(IMAGE) .

run:
	podman run --rm -it \
		-e ANTHROPIC_API_KEY -e OPENAI_API_KEY \
		-v "$(SKILLS):/skills:ro,Z" \
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
