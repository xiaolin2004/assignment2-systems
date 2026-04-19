import torch
import sys

def test_oom():
    BATCH_SIZE = 8
    SEQ_LEN = 16384
    HEAD_DIM = 128 # Largest config
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing OOM on {device} with SeqLen={SEQ_LEN}, HeadDim={HEAD_DIM}...")
    
    try:
        Q = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=device, requires_grad=True)
        K = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=device, requires_grad=True)
        V = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=device, requires_grad=True)
        
        print("Allocated inputs. Running forward...")
        from cs336_systems.model import scaled_dot_product_attention
        out = scaled_dot_product_attention(Q, K, V)
        print("Forward done. Running backward...")
        out.sum().backward()
        print("Backward done. No OOM.")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        if "out of memory" in str(e).lower():
            print("Confirmed OOM.")

if __name__ == "__main__":
    test_oom()
