package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.ERelationshipRequestStatus;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.time.LocalDateTime;

/**
 * Entity lưu trữ yêu cầu kết nối giữa Elder và Supervisor
 * Ai cũng có thể gửi request, nhưng phải được chấp nhận
 */
@Entity
@Table(name = "relationship_requests")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class RelationshipRequest extends BaseEntity {
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "elder_user_id", nullable = false)
    private ElderUser elderUser; // Elder trong relationship
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "supervisor_user_id", nullable = false)
    private SupervisorUser supervisorUser; // Supervisor trong relationship
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "requester_id", nullable = false)
    private User requester; // Người gửi request (có thể là Elder hoặc Supervisor)
    
    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private ERelationshipRequestStatus status = ERelationshipRequestStatus.PENDING;
    
    @Column(name = "message", length = 500)
    private String message; // Lời nhắn khi gửi request
    
    @Column(name = "responded_at")
    private LocalDateTime respondedAt; // Thời điểm accept/reject
    
    @Column(name = "response_message", length = 500)
    private String responseMessage; // Lời nhắn khi respond
}
