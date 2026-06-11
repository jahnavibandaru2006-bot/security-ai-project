from ultralytics import YOLO
import cv2
import time
import numpy as np

print("🔬 AI Model Research & Optimization")
print("=" * 50)

# Models to compare
models_to_test = [
    ('yolov8n.pt', 'YOLOv8 Nano (Smallest)'),
    ('yolov8s.pt', 'YOLOv8 Small'),
]

# Test image — use webcam frame
cap = cv2.VideoCapture(0)
ret, test_frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not capture test frame")
    exit()

print(f"\n📸 Test image size: {test_frame.shape}")
print("\n🚀 Testing models...\n")

results_summary = []

for model_file, model_name in models_to_test:
    print(f"Testing: {model_name}")
    try:
        model = YOLO(model_file)

        # Warmup run
        model(test_frame, verbose=False)

        # Test 10 times and average
        times = []
        detections = 0
        for i in range(10):
            start = time.time()
            results = model(test_frame, verbose=False)
            end = time.time()
            times.append(end - start)
            for r in results:
                detections += len(r.boxes)

        avg_time = np.mean(times)
        avg_detections = detections / 10
        fps = 1 / avg_time

        print(f"  ✅ Average time: {avg_time:.3f}s")
        print(f"  ✅ FPS: {fps:.1f}")
        print(f"  ✅ Avg detections: {avg_detections:.1f}")
        print()

        results_summary.append({
            'name': model_name,
            'file': model_file,
            'avg_time': avg_time,
            'fps': fps,
            'detections': avg_detections
        })

    except Exception as e:
        print(f"  ❌ Error: {e}\n")

# Print final report
print("=" * 50)
print("📊 FINAL COMPARISON REPORT")
print("=" * 50)
print(f"{'Model':<25} {'FPS':<10} {'Avg Time':<12} {'Detections'}")
print("-" * 60)
for r in results_summary:
    print(f"{r['name']:<25} {r['fps']:<10.1f} {r['avg_time']:<12.3f} {r['detections']:.1f}")

print("\n🏆 RECOMMENDATION:")
if results_summary:
    fastest = min(results_summary, key=lambda x: x['avg_time'])
    print(f"Best model for your device: {fastest['name']}")
    print(f"Reason: Fastest at {fastest['fps']:.1f} FPS")

# Save report
with open("../reports/model_comparison.txt", "w") as f:
    f.write("AI Model Comparison Report\n")
    f.write("=" * 50 + "\n\n")
    for r in results_summary:
        f.write(f"Model: {r['name']}\n")
        f.write(f"FPS: {r['fps']:.1f}\n")
        f.write(f"Avg Time: {r['avg_time']:.3f}s\n")
        f.write(f"Detections: {r['detections']:.1f}\n\n")

print("\n📄 Report saved to reports/model_comparison.txt")
print("\n✅ Module 5 Complete!")