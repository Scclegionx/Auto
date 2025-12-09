package com.example.Auto_BE.mapper;

import com.example.Auto_BE.dto.response.EmergencyContactResponse;
import com.example.Auto_BE.entity.EmergencyContact;

public class EmergencyContactMapper {
    
    public static EmergencyContactResponse toResponse(EmergencyContact contact) {
        if (contact == null) {
            return null;
        }
        
        return EmergencyContactResponse.builder()
                .id(contact.getId())
                .name(contact.getName())
                .phoneNumber(contact.getPhoneNumber())
                .address(contact.getAddress())
                .relationship(contact.getRelationship())
                .note(contact.getNote())
                .userId(contact.getElderUser() != null ? contact.getElderUser().getId() : null)
                .createdAt(contact.getCreatedAt())
                .updatedAt(contact.getUpdatedAt())
                .build();
    }
}
