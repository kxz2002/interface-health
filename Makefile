.PHONY: setup install torch-cpu torch-gpu test lint format clean install-skills

# CUDA 版本可按本机驱动修改，常见值: cu118 / cu121 / cu124
CUDA_VERSION ?= cu121

# 一条命令复现环境（不含 torch，装完后按设备运行 make torch-cpu 或 make torch-gpu）
setup:
	conda env create -f environment.yml
	conda run -n interface pip install -e ".[dev]"
	conda run -n interface pre-commit install
	$(MAKE) install-skills
	@echo "环境创建完成。请继续运行:"
	@echo "  CPU 设备:  make torch-cpu"
	@echo "  GPU 设备:  make torch-gpu  (默认 CUDA=$(CUDA_VERSION)，可用 make torch-gpu CUDA_VERSION=cu124 覆盖)"

# 同步环境到 environment.yml 最新状态（已有环境、拉新代码后用）
sync:
	conda env update -n interface -f environment.yml --prune
	conda run -n interface pip install -e ".[dev]"
	@echo "环境已同步。如果新代码新增了 torch 依赖变更，请手动运行 make torch-cpu 或 make torch-gpu"

# 仅安装包（环境已存在时用）
install:
	pip install -e ".[dev]"
	pre-commit install

# torch CPU 版（无 GPU 设备 / CI 环境）
torch-cpu:
	pip install torch --index-url https://download.pytorch.org/whl/cpu

# torch GPU 版，默认 CUDA_VERSION=cu121，可命令行覆盖
torch-gpu:
	pip install torch --index-url https://download.pytorch.org/whl/$(CUDA_VERSION)

test:
	pytest tests/ -v

lint:
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# 将 skills-local/ 下的 project-local skills 链接到 .claude/skills/（新协作者 clone 后运行）
install-skills:
	@mkdir -p .claude/skills
	@if [ ! -d skills-local ] || [ -z "$$(ls -d skills-local/*/ 2>/dev/null)" ]; then \
		echo "  (skills-local/ 为空，无需安装)"; \
		exit 0; \
	fi
	@for skill in skills-local/*/; do \
		[ -d "$$skill" ] || continue; \
		name=$$(basename $$skill); \
		target=".claude/skills/$$name"; \
		if [ -L "$$target" ]; then \
			echo "  exists: $$name (symlink, skipped)"; \
		elif [ -e "$$target" ]; then \
			echo "ERROR: $$target 已存在且不是 symlink，请手动删除后重试" >&2; \
			exit 1; \
		else \
			ln -sf "../../$$skill" "$$target" || { echo "ERROR: 无法创建 symlink: $$name" >&2; exit 1; }; \
			echo "  linked: $$name"; \
		fi \
	done

# 清理实验产物，不触碰数据和模型 checkpoint
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf outputs/.hydra
