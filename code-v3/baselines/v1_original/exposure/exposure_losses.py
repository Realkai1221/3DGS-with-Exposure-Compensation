import torch


def exposure_rgb_loss(I_rendered, I_gt, exposure_map, epsilon=1e-4):

    adjusted_gt = I_gt / (exposure_map + epsilon)
    return torch.nn.functional.mse_loss(I_rendered, adjusted_gt)


def exposure_smooth_loss(exposure_tensor, neighbor_idx):

    if not neighbor_idx or len(neighbor_idx) == 0:
        return torch.tensor(0.0, device=exposure_tensor.device)

    if len(exposure_tensor) > 10000:
        return torch.tensor(0.0, device=exposure_tensor.device)

    device = exposure_tensor.device
    total_loss = torch.tensor(0.0, device=device)

    batch_size = 1000
    for start_idx in range(0, len(neighbor_idx), batch_size):
        end_idx = min(start_idx + batch_size, len(neighbor_idx))
        batch_loss = 0.0

        for i in range(start_idx, end_idx):
            if neighbor_idx[i]:  
                neighbors = torch.tensor(neighbor_idx[i], device=device, dtype=torch.long)
                diff = exposure_tensor[i] - exposure_tensor[neighbors]
                batch_loss += (diff ** 2).sum()

        total_loss += batch_loss

    num_edges = sum(len(n) for n in neighbor_idx)
    return total_loss / (num_edges + 1e-6)


def exposure_range_loss(exposure_tensor):

    lower = torch.relu(0.1 - exposure_tensor)
    upper = torch.relu(exposure_tensor - 1.0)
    return (lower ** 2 + upper ** 2).mean()


def exposure_consistency_loss(exposure_tensor, multi_view_exposures):

    device = exposure_tensor.device

    if not multi_view_exposures or len(multi_view_exposures) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_pairs = 0

    processed = 0
    skipped = 0

    for idx, exp_list in multi_view_exposures.items():
        if not exp_list or len(exp_list) < 2:
            continue

        try:
            valid_exps = []

            for exp in exp_list:
                if isinstance(exp, torch.Tensor):
                    if exp.numel() == 1:
                        val = float(exp.detach().cpu().item())
                    else:
                        continue  
                elif isinstance(exp, (int, float)):
                    val = float(exp)
                elif hasattr(exp, 'item'): 
                    val = float(exp.item())
                else:
                    continue  

                if torch.isfinite(torch.tensor(val)) and 0.01 < val < 2.0:
                    valid_exps.append(val)

            if len(valid_exps) < 2:
                skipped += 1
                continue

            exp_tensor = torch.tensor(valid_exps, device=device, dtype=torch.float32)

            if exp_tensor.numel() < 2:
                skipped += 1
                continue

            var = exp_tensor.var(unbiased=False)

            if not torch.isfinite(var):
                skipped += 1
                continue
                
            total_loss = total_loss + var
            total_pairs += 1
            processed += 1

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"Warning: Error processing exposure consistency for idx {idx}: {type(e).__name__}: {str(e)}")
                if exp_list and len(exp_list) > 0:
                    print(f"  First value type: {type(exp_list[0])}, value: {exp_list[0]}")
            continue

    if processed + skipped > 0 and (processed + skipped) % 100 == 0:
        print(f"Exposure consistency: processed {processed}, skipped {skipped} gaussians")

    if total_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / float(total_pairs)
