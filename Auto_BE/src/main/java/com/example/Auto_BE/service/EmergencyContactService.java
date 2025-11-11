package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.request.EmergencyContactCreateRequest;
import com.example.Auto_BE.dto.request.EmergencyContactUpdateRequest;
import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.response.EmergencyContactResponse;
import com.example.Auto_BE.entity.EmergencyContact;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.mapper.EmergencyContactMapper;
import com.example.Auto_BE.repository.EmergencyContactRepository;
import com.example.Auto_BE.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

import static com.example.Auto_BE.constants.ErrorMessages.USER_NOT_FOUND;
import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;


@Service
@RequiredArgsConstructor
@Slf4j
public class EmergencyContactService {
    
    private final EmergencyContactRepository emergencyContactRepository;
    private final UserRepository userRepository;
    
    /**
     * Tạo liên hệ khẩn cấp mới
     */
    @Transactional
    public BaseResponse<EmergencyContactResponse> create(
            EmergencyContactCreateRequest request, 
            Authentication authentication) {
        try {
            // Lấy user hiện tại
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
            
            // Tạo emergency contact mới
            EmergencyContact contact = new EmergencyContact()
                    .setName(request.getName())
                    .setPhoneNumber(request.getPhoneNumber())
                    .setAddress(request.getAddress())
                    .setRelationship(request.getRelationship())
                    .setNote(request.getNote())
                    .setUser(user);
            
            EmergencyContact savedContact = emergencyContactRepository.save(contact);
            
            log.info("Created emergency contact: {} for user: {}", savedContact.getId(), user.getEmail());
            
            return BaseResponse.<EmergencyContactResponse>builder()
                    .status(SUCCESS)
                    .message("Tạo liên hệ khẩn cấp thành công")
                    .data(EmergencyContactMapper.toResponse(savedContact))
                    .build();
                    
        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            log.error("Error creating emergency contact", e);
            throw new BaseException.BadRequestException("Lỗi khi tạo liên hệ khẩn cấp: " + e.getMessage());
        }
    }
    
    /**
     * Lấy tất cả liên hệ khẩn cấp của user
     */
    public BaseResponse<List<EmergencyContactResponse>> getAllByUser(Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
            
            List<EmergencyContact> contacts = emergencyContactRepository.findByUserOrderByCreatedAtDesc(user);
            
            List<EmergencyContactResponse> responses = contacts.stream()
                    .map(EmergencyContactMapper::toResponse)
                    .collect(Collectors.toList());
            
            return BaseResponse.<List<EmergencyContactResponse>>builder()
                    .status(SUCCESS)
                    .message("Lấy danh sách liên hệ khẩn cấp thành công")
                    .data(responses)
                    .build();
                    
        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            log.error("Error getting emergency contacts", e);
            throw new BaseException.BadRequestException("Lỗi khi lấy danh sách liên hệ khẩn cấp: " + e.getMessage());
        }
    }
    
    /**
     * Lấy chi tiết liên hệ khẩn cấp theo ID
     */
    public BaseResponse<EmergencyContactResponse> getById(Long id, Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
            
            EmergencyContact contact = emergencyContactRepository.findByIdAndUser(id, user)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Không tìm thấy liên hệ khẩn cấp"));
            
            return BaseResponse.<EmergencyContactResponse>builder()
                    .status(SUCCESS)
                    .message("Lấy thông tin liên hệ khẩn cấp thành công")
                    .data(EmergencyContactMapper.toResponse(contact))
                    .build();
                    
        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            log.error("Error getting emergency contact by id: {}", id, e);
            throw new BaseException.BadRequestException("Lỗi khi lấy thông tin liên hệ khẩn cấp: " + e.getMessage());
        }
    }
    
    /**
     * Cập nhật liên hệ khẩn cấp
     */
    @Transactional
    public BaseResponse<EmergencyContactResponse> update(
            Long id,
            EmergencyContactUpdateRequest request,
            Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
            
            EmergencyContact contact = emergencyContactRepository.findByIdAndUser(id, user)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Không tìm thấy liên hệ khẩn cấp"));
            
            // Update fields
            contact.setName(request.getName())
                   .setPhoneNumber(request.getPhoneNumber())
                   .setAddress(request.getAddress())
                   .setRelationship(request.getRelationship())
                   .setNote(request.getNote());
            
            EmergencyContact updatedContact = emergencyContactRepository.save(contact);
            
            log.info("Updated emergency contact: {} for user: {}", id, user.getEmail());
            
            return BaseResponse.<EmergencyContactResponse>builder()
                    .status(SUCCESS)
                    .message("Cập nhật liên hệ khẩn cấp thành công")
                    .data(EmergencyContactMapper.toResponse(updatedContact))
                    .build();
                    
        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            log.error("Error updating emergency contact: {}", id, e);
            throw new BaseException.BadRequestException("Lỗi khi cập nhật liên hệ khẩn cấp: " + e.getMessage());
        }
    }
    
    /**
     * Xóa liên hệ khẩn cấp
     */
    @Transactional
    public BaseResponse<Void> delete(Long id, Authentication authentication) {
        try {
            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
            
            EmergencyContact contact = emergencyContactRepository.findByIdAndUser(id, user)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Không tìm thấy liên hệ khẩn cấp"));
            
            emergencyContactRepository.delete(contact);
            
            log.info("Deleted emergency contact: {} for user: {}", id, user.getEmail());
            
            return BaseResponse.<Void>builder()
                    .status(SUCCESS)
                    .message("Xóa liên hệ khẩn cấp thành công")
                    .build();
                    
        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            log.error("Error deleting emergency contact: {}", id, e);
            throw new BaseException.BadRequestException("Lỗi khi xóa liên hệ khẩn cấp: " + e.getMessage());
        }
    }
}
