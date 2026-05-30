"""Run every assignment in order. Usage: python run_all.py"""
import runpy
for name in ["kmeans_assignment1", "kmeans_assignment2", "kmeans_assignment3",
             "gmm_assignment1"]:
    print("\n" + "=" * 72 + f"\nRUNNING {name}.py\n" + "=" * 72)
    runpy.run_module(name, run_name="__main__")
