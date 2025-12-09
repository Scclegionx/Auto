package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.ERelationshipRequestStatus;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.time.Instant;

@Entity
@Table(name = "elder_supervisor")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class ElderSupervisor extends BaseEntity {
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "elder_user_id", nullable = false)
    private ElderUser elderUser;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "supervisor_user_id", nullable = false)
    private SupervisorUser supervisorUser;
    
    // ===== Request/Response Flow =====
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "requester_id", nullable = false)
    private User requester; // Người gửi request (Elder hoặc Supervisor)
    
    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private ERelationshipRequestStatus status = ERelationshipRequestStatus.PENDING;
    
    @Column(name = "request_message", length = 500)
    private String requestMessage; // Lời nhắn khi gửi request
    
    @Column(name = "responded_at")
    private Instant respondedAt; // Thời điểm accept/reject
    
    @Column(name = "response_message", length = 500)
    private String responseMessage; // Lời nhắn khi respond
    
    // ===== Permissions (chỉ active khi status = ACCEPTED) =====
    
    @Column(name = "can_view_prescription", nullable = false)
    private Boolean canViewPrescription = true; // Default true khi accept
    
    @Column(name = "can_update_prescription", nullable = false)
    private Boolean canUpdatePrescription = false; // Default false, có thể cấp sau
    
    @Column(name = "is_active")
    private Boolean isActive = true; // true khi ACCEPTED, false khi bị xóa/vô hiệu hóa
    
    @Column(name = "note")
    private String note; // Ghi chú về mối quan hệ giám sát
}
