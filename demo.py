"""MobileSAM demo + benchmark. Run from the MobileSAM repo folder:  python demo.py
Put your own photos into the 'images' folder (optional). Results go to 'results'."""
import os, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
sam.to(device).eval()
pred = SamPredictor(sam)

os.makedirs("results", exist_ok=True)
paths = sorted(glob.glob("images/*.jpg") + glob.glob("images/*.png")) or sorted(glob.glob("app/assets/*.jpg"))[:3]

def sync():
    if device == "cuda":
        torch.cuda.synchronize()

N = 5
rows = []
for path in paths:
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    pt = np.array([[W // 2, H // 2]])
    pred.set_image(img)                      # warm-up (not counted)
    te, td = [], []
    for _ in range(N):
        sync(); t = time.perf_counter(); pred.set_image(img); sync(); te.append(time.perf_counter() - t)
        sync(); t = time.perf_counter()
        masks, scores, _ = pred.predict(point_coords=pt, point_labels=np.array([1]), multimask_output=True)
        sync(); td.append(time.perf_counter() - t)
    e, d = np.mean(te) * 1000, np.mean(td) * 1000
    rows.append((name, f"{W}x{H}", e, d, scores.max()))
    print(f"{name} {W}x{H}: encoder {e:.1f} ms, decoder {d:.1f} ms, best score {scores.max():.3f}")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img); axs[0].plot(*pt[0], "r*", ms=15); axs[0].set_title("Input + point")
    for i in range(3):
        axs[i + 1].imshow(img)
        axs[i + 1].imshow(masks[i], alpha=0.5, cmap="Blues")
        axs[i + 1].set_title(f"Mask {i+1}, score={scores[i]:.3f}")
    for a in axs: a.axis("off")
    plt.tight_layout(); plt.savefig(f"results/{name}_point.png", dpi=80); plt.close()

print("\nSummary (mean of", N, "runs):")
for r in rows:
    print(f"{r[0]:15s} {r[1]:10s} encoder {r[2]:8.1f} ms   decoder {r[3]:7.1f} ms")

# automatic mode on the first image
img = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
gen = SamAutomaticMaskGenerator(sam, points_per_side=16)
t = time.perf_counter(); anns = gen.generate(img); dt = time.perf_counter() - t
print(f"\nAuto mode: {len(anns)} masks in {dt:.1f} s")
out = img.astype(float); rng = np.random.default_rng(0)
for a in sorted(anns, key=lambda x: -x["area"]):
    out[a["segmentation"]] = 0.45 * out[a["segmentation"]] + 0.55 * rng.integers(0, 255, 3)
cv2.imwrite("results/auto.png", cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR))
