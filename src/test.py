import torch
coords = torch.load("my_motion.pt")

# Check hand vs knee positions in first frame
frame = coords[0]
left_hand  = frame[20]  # left_wrist
left_knee  = frame[4]   # left_knee

print("left_wrist:", left_hand)
print("left_knee: ", left_knee)
print("distance:  ", (left_hand - left_knee).norm().item())