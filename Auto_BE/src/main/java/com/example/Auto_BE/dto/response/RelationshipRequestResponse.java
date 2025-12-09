package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.EBloodType;
import com.example.Auto_BE.entity.enums.EGender;
import com.example.Auto_BE.entity.enums.ERelationshipRequestStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RelationshipRequestResponse {
    
    private Long id;
    
    private Long elderUserId;
    private String elderUserName;
    private String elderUserEmail;
    private String elderUserAvatar;
    private String elderUserPhone;
    private EBloodType elderBloodType;
    private Double elderHeight;
    private Double elderWeight;
    private EGender elderGender;
    
    private Long supervisorUserId;
    private String supervisorUserName;
    private String supervisorUserEmail;
    private String supervisorUserAvatar;
    private String supervisorOccupation;
    private String supervisorWorkplace;
    
    private Long requesterId; // Người gửi request
    private String requesterName;
    
    private ERelationshipRequestStatus status;
    
    private String requestMessage; // Lời nhắn khi gửi
    private String responseMessage; // Lời nhắn khi accept/reject
    
    private Instant respondedAt;
    private Instant createdAt;
}
