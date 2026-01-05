package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.CreateUserGuideRequest;
import com.example.Auto_BE.dto.request.UpdateUserGuideRequest;
import com.example.Auto_BE.dto.response.UserGuideResponse;
import com.example.Auto_BE.entity.UserGuide;
import com.example.Auto_BE.entity.enums.EUserType;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.UserGuideRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional
public class UserGuideService {

    @Autowired
    private UserGuideRepository userGuideRepository;
    
    @Autowired
    private CloudinaryService cloudinaryService;

    /**
     * Tạo user guide mới (có upload video)
     */
    public BaseResponse<UserGuideResponse> createUserGuide(
            CreateUserGuideRequest request, 
            MultipartFile videoFile) {
        try {
            // Validate video file
            if (videoFile == null || videoFile.isEmpty()) {
                return BaseResponse.<UserGuideResponse>builder()
                        .status("error")
                        .message("Video file is required")
                        .data(null)
                        .build();
            }

            // Upload video lên Cloudinary
            String videoUrl;
            try {
                videoUrl = cloudinaryService.uploadVideo(videoFile);
            } catch (IOException e) {
                return BaseResponse.<UserGuideResponse>builder()
                        .status("error")
                        .message("Failed to upload video: " + e.getMessage())
                        .data(null)
                        .build();
            }

            // Tạo UserGuide entity
            UserGuide userGuide = new UserGuide();
            userGuide.setTitle(request.getTitle());
            userGuide.setDescription(request.getDescription());
            userGuide.setVideoUrl(videoUrl);
            userGuide.setThumbnailUrl(request.getThumbnailUrl());
            userGuide.setUserType(request.getUserType());
            userGuide.setDisplayOrder(request.getDisplayOrder() != null ? request.getDisplayOrder() : 0);

            // Save
            UserGuide saved = userGuideRepository.save(userGuide);

            // Convert to response
            UserGuideResponse response = convertToResponse(saved);

            return BaseResponse.<UserGuideResponse>builder()
                    .status("success")
                    .message("Tạo hướng dẫn sử dụng thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            System.err.println("Error creating user guide: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<UserGuideResponse>builder()
                    .status("error")
                    .message("Lỗi khi tạo hướng dẫn sử dụng: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy user guide theo ID
     */
    public BaseResponse<UserGuideResponse> getUserGuideById(Long id) {
        try {
            UserGuide userGuide = userGuideRepository.findById(id)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Hướng dẫn sử dụng không tồn tại"));

            UserGuideResponse response = convertToResponse(userGuide);

            return BaseResponse.<UserGuideResponse>builder()
                    .status("success")
                    .message("Lấy hướng dẫn sử dụng thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            return BaseResponse.<UserGuideResponse>builder()
                    .status("error")
                    .message("Lỗi khi lấy hướng dẫn sử dụng: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy tất cả user guides của Elder
     */
    public BaseResponse<List<UserGuideResponse>> getElderUserGuides() {
        try {
            List<UserGuide> userGuides = userGuideRepository
                    .findByUserTypeOrderByDisplayOrderAsc(EUserType.ELDER);

            List<UserGuideResponse> responseList = userGuides.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<UserGuideResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách hướng dẫn sử dụng cho Elder thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            return BaseResponse.<List<UserGuideResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách hướng dẫn sử dụng: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy tất cả user guides của Supervisor
     */
    public BaseResponse<List<UserGuideResponse>> getSupervisorUserGuides() {
        try {
            List<UserGuide> userGuides = userGuideRepository
                    .findByUserTypeOrderByDisplayOrderAsc(EUserType.SUPERVISOR);

            List<UserGuideResponse> responseList = userGuides.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<UserGuideResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách hướng dẫn sử dụng cho Supervisor thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            return BaseResponse.<List<UserGuideResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách hướng dẫn sử dụng: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Update user guide
     */
    public BaseResponse<UserGuideResponse> updateUserGuide(
            Long id, 
            UpdateUserGuideRequest request,
            MultipartFile videoFile) {
        try {
            UserGuide userGuide = userGuideRepository.findById(id)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Hướng dẫn sử dụng không tồn tại"));

            // Update fields
            if (request.getTitle() != null) {
                userGuide.setTitle(request.getTitle());
            }
            if (request.getDescription() != null) {
                userGuide.setDescription(request.getDescription());
            }
            if (request.getUserType() != null) {
                userGuide.setUserType(request.getUserType());
            }
            if (request.getThumbnailUrl() != null) {
                userGuide.setThumbnailUrl(request.getThumbnailUrl());
            }
            if (request.getDisplayOrder() != null) {
                userGuide.setDisplayOrder(request.getDisplayOrder());
            }

            // Nếu có video file mới, upload và xóa video cũ
            if (videoFile != null && !videoFile.isEmpty()) {
                try {
                    // Xóa video cũ
                    if (userGuide.getVideoUrl() != null) {
                        cloudinaryService.deleteVideo(userGuide.getVideoUrl());
                    }
                    // Upload video mới
                    String newVideoUrl = cloudinaryService.uploadVideo(videoFile);
                    userGuide.setVideoUrl(newVideoUrl);
                } catch (IOException e) {
                    return BaseResponse.<UserGuideResponse>builder()
                            .status("error")
                            .message("Failed to upload new video: " + e.getMessage())
                            .data(null)
                            .build();
                }
            } else if (request.getVideoUrl() != null) {
                // Nếu chỉ update URL (trường hợp đặc biệt)
                userGuide.setVideoUrl(request.getVideoUrl());
            }

            // Save
            UserGuide updated = userGuideRepository.save(userGuide);

            // Convert to response
            UserGuideResponse response = convertToResponse(updated);

            return BaseResponse.<UserGuideResponse>builder()
                    .status("success")
                    .message("Cập nhật hướng dẫn sử dụng thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            System.err.println("Error updating user guide: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<UserGuideResponse>builder()
                    .status("error")
                    .message("Lỗi khi cập nhật hướng dẫn sử dụng: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Xóa user guide
     */
    public BaseResponse<String> deleteUserGuide(Long id) {
        try {
            UserGuide userGuide = userGuideRepository.findById(id)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Hướng dẫn sử dụng không tồn tại"));

            // Xóa video từ Cloudinary
            if (userGuide.getVideoUrl() != null) {
                cloudinaryService.deleteVideo(userGuide.getVideoUrl());
            }

            // Xóa từ database
            userGuideRepository.delete(userGuide);

            return BaseResponse.<String>builder()
                    .status("success")
                    .message("Xóa hướng dẫn sử dụng thành công")
                    .data("User guide ID " + id + " đã được xóa")
                    .build();

        } catch (Exception e) {
            System.err.println("Error deleting user guide: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<String>builder()
                    .status("error")
                    .message("Lỗi khi xóa hướng dẫn sử dụng: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    // ===== HELPER METHODS =====

    private UserGuideResponse convertToResponse(UserGuide userGuide) {
        return UserGuideResponse.builder()
                .id(userGuide.getId())
                .title(userGuide.getTitle())
                .description(userGuide.getDescription())
                .videoUrl(userGuide.getVideoUrl())
                .thumbnailUrl(userGuide.getThumbnailUrl())
                .userType(userGuide.getUserType())
                .displayOrder(userGuide.getDisplayOrder())
                .createdAt(userGuide.getCreatedAt() != null ? 
                    java.time.LocalDateTime.ofInstant(userGuide.getCreatedAt(), java.time.ZoneId.systemDefault()) : null)
                .updatedAt(userGuide.getUpdatedAt() != null ? 
                    java.time.LocalDateTime.ofInstant(userGuide.getUpdatedAt(), java.time.ZoneId.systemDefault()) : null)
                .build();
    }
}

