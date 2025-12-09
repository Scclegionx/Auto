package com.example.Auto_BE.mapper;

import com.example.Auto_BE.dto.response.ProfileResponse;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.entity.User;

public class UserMapper {
    public static ProfileResponse toResponse(User user){
        ProfileResponse.ProfileResponseBuilder builder = ProfileResponse.builder()
                .id(user.getId())
                .fullName(user.getFullName())
                .email(user.getEmail())
                .dateOfBirth(user.getDateOfBirth())
                .gender(user.getGender())
                .phoneNumber(user.getPhoneNumber())
                .address(user.getAddress())
                .avatar(user.getAvatar())
                .isActive(user.getIsActive());
        
        // Check nếu là ElderUser thì thêm thông tin sức khỏe
        if (user instanceof ElderUser) {
            ElderUser elderUser = (ElderUser) user;
            builder
                .role("ELDER")
                .bloodType(elderUser.getBloodType())
                .height(elderUser.getHeight())
                .weight(elderUser.getWeight());
        }
        // Check nếu là SupervisorUser thì thêm thông tin công việc
        else if (user instanceof SupervisorUser) {
            SupervisorUser supervisorUser = (SupervisorUser) user;
            builder
                .role("SUPERVISOR")
                .occupation(supervisorUser.getOccupation())
                .workplace(supervisorUser.getWorkplace());
        }
        else {
            builder.role("USER");
        }
        
        return builder.build();
    }


}