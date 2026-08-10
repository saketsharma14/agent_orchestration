from orchestrator.loop import run_search

if __name__ == "__main__":
    archive = run_search()
    best = archive.top_k(1)
    if best:
        print("Best candidate found:")
        print(best[0].candidate.to_json())
        print(f"fitness={best[0].fitness_scalar:.4f}")
    else:
        print("No valid candidates found.")