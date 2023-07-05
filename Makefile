ROOT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
FILES := $(shell find $(ROOT_DIR)oncofem/ -name "*.py")
OUTPUT_DIR := $(ROOT_DIR)doc

print_location:
	@echo "Location is $(ROOT_DIR)doc"

docs: $(FILES)
	@mkdir -p $(OUTPUT_DIR)
	@for file in $(FILES); do \
		pydoc -w "$$file"; \
		filepath=$$(dirname "$$file" | sed 's|$(ROOT_DIR)||'); \
		filename=$$(basename "$$file" .py); \
		subdir=$$(echo "$$filepath" | tr '/' '.'); \
		mv "$$filename.html" $(OUTPUT_DIR)/"$$subdir.$$filename.html"; \
	done

clean:
	@rm -fr $(OUTPUT_DIR)

