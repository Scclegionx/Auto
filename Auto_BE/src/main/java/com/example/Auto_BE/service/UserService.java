package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.ChangePasswordRequest;
import com.example.Auto_BE.dto.request.UpdateProfileRequest;
import com.example.Auto_BE.dto.response.ProfileResponse;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.EGender;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.mapper.UserMapper;
import com.example.Auto_BE.repository.UserRepository;
import com.google.firebase.remoteconfig.internal.TemplateResponse;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.LocalDate;

import static com.example.Auto_BE.constants.ErrorMessages.*;
import static com.example.Auto_BE.constants.SuccessMessage.*;

@Service
public class UserService {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public BaseResponse<?> getUserProfile(Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
        ProfileResponse profileResponse= UserMapper.toResponse(user);
        return BaseResponse.<ProfileResponse>builder()
                .status(SUCCESS)
                .message(PROFILE_FETCHED)
                .data(profileResponse)
                .build();
    }

    public BaseResponse<?> updateUserProfile(UpdateProfileRequest updateProfileRequest, Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
            if(updateProfileRequest.getFullName() != null) {
                user.setFullName(updateProfileRequest.getFullName());
            }
            if(updateProfileRequest.getPhoneNumber()!= null) {
                user.setPhoneNumber(updateProfileRequest.getPhoneNumber());
            }
            if (updateProfileRequest.getDateOfBirth() != null) {
                user.setDateOfBirth(updateProfileRequest.getDateOfBirth());
            }

            if (updateProfileRequest.getGender() != null) {
                user.setGender(updateProfileRequest.getGender());
            }

            if (updateProfileRequest.getAddress() != null) {
                user.setAddress(updateProfileRequest.getAddress());
            }

            if (updateProfileRequest.getBloodType() != null) {
                user.setBloodType(updateProfileRequest.getBloodType());
            }

            if (updateProfileRequest.getHeight() != null) {
                user.setHeight(updateProfileRequest.getHeight());
            }

            if (updateProfileRequest.getWeight() != null) {
                user.setWeight(updateProfileRequest.getWeight());
            }

            User updatedUser = userRepository.save(user);
            ProfileResponse profileResponse = UserMapper.toResponse(updatedUser);
            return BaseResponse.<ProfileResponse>builder()
                    .status(SUCCESS)
                    .message(PROFILE_UPDATED)
                    .data(profileResponse)
                    .build();


        }catch (Exception e) {
            throw new BaseException.BadRequestException(e.getMessage());
        }
    }

    public BaseResponse<String> changePassword(ChangePasswordRequest changePasswordRequest, Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

            if (!passwordEncoder.matches(changePasswordRequest.getCurrentPassword(), user.getPassword())) {
                throw new BaseException.BadRequestException(CURR_PASSWORD_INCORRECT);
            }

            if (passwordEncoder.matches(changePasswordRequest.getNewPassword(), user.getPassword())) {
                throw new BaseException.BadRequestException(PASSWORD_ERROR);
            }

            user.setPassword(passwordEncoder.encode(changePasswordRequest.getNewPassword()));
            userRepository.save(user);

            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message(PASSWORD_UPDATED)
                    .data(null)
                    .build();

        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            throw new BaseException.BadRequestException(e.getMessage());
        }
    }


}