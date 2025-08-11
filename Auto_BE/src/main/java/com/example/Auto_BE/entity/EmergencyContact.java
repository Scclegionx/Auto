package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

@Entity
@Table(name = "emergency_contacts")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class EmergencyContact extends BaseEntity{
    @Column(name = "name", nullable = false)
    private String name; // Tên người liên hệ khẩn cấp

    @Column(name = "phone_number", nullable = false)
    private String phoneNumber; // Số điện thoại người liên hệ khẩn cấp

    @Column(name = "address")
    private String address; // Địa chỉ người liên hệ khẩn cấp (nếu cần)

    @Column(name = "relationship", nullable = false)
    private String relationship; // Mối quan hệ với người liên hệ khẩn cấp (ví dụ: Bố, Mẹ, Bạn bè, v.v.)

    @Column(name = "note")
    private String note; // Ghi chú thêm về người liên hệ khẩn cấp (nếu cần)

    @ManyToOne(fetch = FetchType.LAZY)
    private User user; // Người dùng sở hữu liên hệ khẩn cấp này
}
