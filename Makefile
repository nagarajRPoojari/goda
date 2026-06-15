.PHONY: merge

# Configuration
BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)

merge:
	@# 1. Validate inputs
	@if [ -z "$(FILES_REGEX)" ]; then echo "Error: FILES_REGEX is required (e.g., FILES_REGEX='\.js$$')"; exit 1; fi
	@if [ -z "$(MSG)" ]; then echo "Error: MSG (commit message) is required"; exit 1; fi
	@if [ -z "$(TYPE)" ]; then echo "Error: TYPE is required (fix, feat, or exp)"; exit 1; fi

	@if [ "$(BRANCH_NAME)" = "main" ] || [ "$(BRANCH_NAME)" = "master" ]; then \
		echo "Error: You cannot run this directly from $(BRANCH_NAME). Switch to a feature branch first."; exit 1; \
	fi

	@echo "--- Starting Merge Pipeline for [$(TYPE)] ---"

	@# 2. Find and add files matching the regex
	@echo "Staging files matching regex: '$(FILES_REGEX)'..."
	@git status --porcelain | awk '{print $$2}' | grep -E "$(FILES_REGEX)" | xargs -r git add
	
	@# 3. Commit and Push
	@echo "Committing changes..."
	@git commit -m "[$(TYPE)] $(MSG)"
	@echo "Pushing branch $(BRANCH_NAME) to remote..."
	@git push origin $(BRANCH_NAME)

	@# 4. Create GitHub PR and Merge it
	@echo "Creating PR: '[$(TYPE)] $(MSG)'..."
	@pr_url=$$(gh pr create --title "[$(TYPE)] $(MSG)" --body "Automated PR via Makefile" --base main --head $(BRANCH_NAME)); \
	echo "PR Created: $$pr_url"; \
	echo "Merging PR into main..."; \

	@# 5. Clean up local state
	@echo "Switching back to main and pulling updates..."
	@git checkout main
	@git pull origin main
	@echo "--- Successfully merged and sync'd! ---"