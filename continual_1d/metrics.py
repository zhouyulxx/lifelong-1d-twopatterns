"""Global 4-way accuracy is the main score; task-aware scores are diagnostic."""
import numpy as np
import torch

@torch.no_grad()
def evaluate(model, x, y, allowed_classes, device, batch_size=256):
    if not len(y):
        raise ValueError("No test examples")
    model.eval()
    all_logits = []
    for offset in range(0, len(y), batch_size):
        inp = torch.from_numpy(x[offset:offset+batch_size]).to(device)
        all_logits.append(model(inp).cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    predicted = np.argmax(logits, axis=1)
    allowed = np.asarray(allowed_classes, dtype=np.int64)
    task_predicted = allowed[np.argmax(logits[:, allowed], axis=1)]
    return {"global_accuracy": float(np.mean(predicted == y)),
            "task_aware_accuracy_diagnostic": float(np.mean(task_predicted == y)),
            "predicted": predicted}
