package com.example.Auto_BE.mapper;

import com.example.Auto_BE.dto.response.ProfileResponse;
import com.example.Auto_BE.entity.User;

public class UserMapper {
    public static ProfileResponse toResponse(User user){
        return ProfileResponse.builder()
                .id(user.getId())
                .fullName(user.getFullName())
                .email(user.getEmail())
                .dateOfBirth(user.getDateOfBirth())
                .gender(user.getGender())
                .phoneNumber(user.getPhoneNumber())
                .address(user.getAddress())
                .bloodType(user.getBloodType())
                .height(user.getHeight())
                .weight(user.getWeight())
                .avatar(user.getAvatar())
                .isActive(user.getIsActive())
                .build();
    }


}
