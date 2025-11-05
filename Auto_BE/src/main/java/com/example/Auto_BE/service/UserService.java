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
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;

import static com.example.Auto_BE.constants.ErrorMessages.*;
import static com.example.Auto_BE.constants.SuccessMessage.*;

@Service
@Slf4j
public class UserService {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final CloudinaryService cloudinaryService;

    private static final List<String> ALLOWED_IMAGE_TYPES = Arrays.asList(
            "image/jpeg", "image/jpg", "image/png", "image/webp",
            "image/gif", "image/apng"  // Hỗ trợ ảnh động
    );
    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB (ảnh động nặng hơn)

    public UserService(UserRepository userRepository, 
                      PasswordEncoder passwordEncoder,
                      CloudinaryService cloudinaryService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.cloudinaryService = cloudinaryService;
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

    public BaseResponse<String> uploadAvatar(MultipartFile avatar, Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

            // Validate file không rỗng
            if (avatar.isEmpty()) {
                throw new BaseException.BadRequestException("File ảnh không được để trống");
            }

            // Validate kích thước file
            if (avatar.getSize() > MAX_FILE_SIZE) {
                throw new BaseException.BadRequestException("Kích thước file không được vượt quá 10MB");
            }

            // Validate loại file
            String contentType = avatar.getContentType();
            if (contentType == null || !ALLOWED_IMAGE_TYPES.contains(contentType.toLowerCase())) {
                throw new BaseException.BadRequestException("Chỉ chấp nhận file ảnh định dạng JPG, PNG, WEBP, GIF");
            }

            // Xóa avatar cũ nếu có
            if (user.getAvatar() != null && !user.getAvatar().isEmpty()) {
                try {
                    cloudinaryService.deleteImage(user.getAvatar());
                    log.info("Deleted old avatar for user: {}", user.getEmail());
                } catch (Exception e) {
                    log.warn("Failed to delete old avatar: {}", e.getMessage());
                    // Tiếp tục upload ảnh mới
                }
            }

            // Upload ảnh mới lên Cloudinary
            String avatarUrl = cloudinaryService.uploadImage(avatar);
            log.info("Uploaded new avatar for user: {} - URL: {}", user.getEmail(), avatarUrl);

            // Cập nhật avatar vào database
            user.setAvatar(avatarUrl);
            userRepository.save(user);

            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Cập nhật ảnh đại diện thành công")
                    .data(avatarUrl)
                    .build();

        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            log.error("Error uploading avatar", e);
            throw new BaseException.BadRequestException("Lỗi khi tải lên ảnh đại diện: " + e.getMessage());
        }
    }

}