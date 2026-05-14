from shadow_detector import full_analysis

tests = [
    ("What is the solar system?", "The solar system consists of the Sun and the objects that orbit it, including eight planets, dwarf planets, asteroids, and comets. It formed approximately 4.6 billion years ago."),
    ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."),
    ("What is CRISPR?", "CRISPR is a gene editing technology that allows scientists to modify DNA sequences with precision. It uses a guide RNA to direct the Cas9 protein to a specific location in the genome where it can cut and edit the DNA."),
]

for prompt, response in tests:
    r = full_analysis(prompt, response)
    print(f"\n--- {prompt[:40]} ---")
    for t in r["tokens"]:
        if t["conflict_status"] != "OK" or t["entropy"] > 4.0:
            print(f"  {t['token']:15} e={t['entropy']:5.2f} z={t['entropy_z_score']:+5.2f} c={t['conflict_score']:.3f} {t['conflict_status']}")
