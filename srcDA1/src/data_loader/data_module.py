from torch.utils.data import DataLoader
from .bearing_dataset import BearingDataset
import os

def create_dataloaders(data_root, batch_size=32, num_workers=2):
    """
    Khởi tạo và trả về Train, Validation, Test DataLoaders.
    
    Args:
        data_root (str): Đường dẫn đến thư mục gốc chứa các tập split (ví dụ: 'data/split').
        batch_size (int): Kích thước lô dữ liệu.
        num_workers (int): Số lượng tiến trình con để tải dữ liệu. 
                           Nên để là 2 để tránh lỗi Slowness/Freeze trên Colab.
    """
    
    # 1. Khởi tạo Datasets cho từng tập split
    # Logic bên trong BearingDataset sẽ tự động duyệt qua tất cả 400 file .pt 
    # trong mỗi folder con của từng tập split tương ứng.
    train_dataset = BearingDataset(data_root=data_root, split='train')
    val_dataset = BearingDataset(data_root=data_root, split='val')
    test_dataset = BearingDataset(data_root=data_root, split='test')
    
    # 2. Khởi tạo DataLoaders
    # Pin_memory=True giúp tăng tốc độ chuyển dữ liệu từ RAM CPU lên VRAM GPU.
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Xáo trộn dữ liệu quan trọng cho quá trình học (Training)
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # Không cần xáo trộn khi đánh giá (Validation)
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # Không cần xáo trộn khi kiểm thử (Test)
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader