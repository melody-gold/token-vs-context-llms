.PHONY: final-report final-report-clean run-70m run-160m probe-70m-splits probe-160m-splits figures-70m figures-160m figures-comparison

PYTHIA70M_CONFIG := configs/pythia70m.yaml
PYTHIA70M_NAME := pythia_pile10k
PYTHIA70M_TITLE := Pythia-70M, Pile-10k

PYTHIA160M_CONFIG := configs/pythia160m.yaml
PYTHIA160M_NAME := pythia_160m_pile10k_50k
PYTHIA160M_TITLE := Pythia-160M, Pile-10k
SPLIT_SEEDS := 0 1 2 3 4 5 6 7 8 9

final-report:
	cd docs && pdflatex final_report/final_report.tex && bibtex final_report && pdflatex final_report/final_report.tex && pdflatex final_report/final_report.tex
	$(MAKE) final-report-clean

final-report-clean:
	rm -f docs/final_report.aux docs/final_report.bbl docs/final_report.bcf docs/final_report.blg docs/final_report.log docs/final_report.out docs/final_report.run.xml docs/final_report.toc docs/missfont.log

run-70m:
	uv run --no-sync token-vs-context extract --config "$(PYTHIA70M_CONFIG)"
	uv run --no-sync token-vs-context probe --config "$(PYTHIA70M_CONFIG)"
	$(MAKE) figures-70m

probe-70m-splits:
	uv run --no-sync token-vs-context probe --config "$(PYTHIA70M_CONFIG)" --random-seeds $(SPLIT_SEEDS)
	$(MAKE) figures-70m

figures-70m:
	uv run --no-sync token-vs-context summarize --metrics "results/generated/$(PYTHIA70M_NAME)_metrics.json" --output "results/generated/$(PYTHIA70M_NAME)_summary.md" --title "$(PYTHIA70M_TITLE) Metrics"
	uv run --no-sync token-vs-context plot --metrics "results/generated/$(PYTHIA70M_NAME)_metrics.json" --output "docs/final_report/figures/$(PYTHIA70M_NAME)_metrics.png" --title "$(PYTHIA70M_TITLE)"
	uv run --no-sync token-vs-context diagnose --config "$(PYTHIA70M_CONFIG)" --output-dir "docs/final_report/figures/$(PYTHIA70M_NAME)_diagnostics" --title "$(PYTHIA70M_TITLE)"

run-160m:
	uv run --no-sync token-vs-context extract --config "$(PYTHIA160M_CONFIG)"
	uv run --no-sync token-vs-context probe --config "$(PYTHIA160M_CONFIG)"
	$(MAKE) figures-160m

probe-160m-splits:
	uv run --no-sync token-vs-context probe --config "$(PYTHIA160M_CONFIG)" --random-seeds $(SPLIT_SEEDS)
	$(MAKE) figures-160m

figures-160m:
	uv run --no-sync token-vs-context summarize --metrics "results/generated/$(PYTHIA160M_NAME)_metrics.json" --output "results/generated/$(PYTHIA160M_NAME)_summary.md" --title "$(PYTHIA160M_TITLE) Metrics"
	uv run --no-sync token-vs-context plot --metrics "results/generated/$(PYTHIA160M_NAME)_metrics.json" --output "docs/final_report/figures/$(PYTHIA160M_NAME)_metrics.png" --title "$(PYTHIA160M_TITLE)"
	uv run --no-sync token-vs-context diagnose --config "$(PYTHIA160M_CONFIG)" --output-dir "docs/final_report/figures/$(PYTHIA160M_NAME)_diagnostics" --title "$(PYTHIA160M_TITLE)"

figures-comparison:
	uv run --no-sync token-vs-context compare --metrics "results/generated/$(PYTHIA70M_NAME)_metrics.json" "results/generated/$(PYTHIA160M_NAME)_metrics.json" --labels "Pythia-70M" "Pythia-160M" --output "docs/final_report/figures/pythia_r2_model_comparison.png" --title 'Token-Only $$R^2$$ Across Model Depth'
