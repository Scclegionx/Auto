package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.DeviceToken;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DeviceTokenRepository extends JpaRepository<DeviceToken, Long> {

    // Các phương thức truy vấn sẽ được định nghĩa ở đây
    // Ví dụ: tìm kiếm theo user, fcmToken, deviceId, v.v.

    // Bạn có thể thêm các phương thức tùy chỉnh nếu cần thiết
}