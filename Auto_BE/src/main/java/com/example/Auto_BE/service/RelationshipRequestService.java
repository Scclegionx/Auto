package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.request.SendRelationshipRequestDTO;
import com.example.Auto_BE.dto.request.RespondToRequestDTO;
import com.example.Auto_BE.dto.response.RelationshipRequestResponse;
import com.example.Auto_BE.dto.response.SupervisorPermissionResponse;
import com.example.Auto_BE.entity.ElderSupervisor;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.ENotificationType;
import com.example.Auto_BE.entity.enums.ERelationshipRequestStatus;
import com.example.Auto_BE.exception.ResourceNotFoundException;
import com.example.Auto_BE.repository.ElderSupervisorRepository;
import com.example.Auto_BE.repository.NotificationRepository;
import com.example.Auto_BE.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class RelationshipRequestService {

    private final ElderSupervisorRepository elderSupervisorRepository;
    private final UserRepository userRepository;
    private final NotificationRepository notificationRepository;

    /**
     * Gửi request kết nối giữa Elder và Supervisor
     */
    @Transactional
    public RelationshipRequestResponse sendRequest(Long requesterId, SendRelationshipRequestDTO requestDTO) {
        User requester = userRepository.findById(requesterId)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng"));

        User target = userRepository.findById(requestDTO.getTargetUserId())
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người nhận"));

        // Kiểm tra không thể gửi request cho chính mình
        if (requesterId.equals(requestDTO.getTargetUserId())) {
            throw new IllegalArgumentException("Không thể gửi yêu cầu cho chính mình");
        }

        // Xác định Elder và Supervisor bằng instanceof
        ElderUser elderUser;
        SupervisorUser supervisorUser;
        
        if (requester instanceof ElderUser && target instanceof SupervisorUser) {
            elderUser = (ElderUser) requester;
            supervisorUser = (SupervisorUser) target;
        } else if (requester instanceof SupervisorUser && target instanceof ElderUser) {
            elderUser = (ElderUser) target;
            supervisorUser = (SupervisorUser) requester;
        } else {
            throw new IllegalArgumentException("Chỉ có thể kết nối giữa Elder và Supervisor");
        }

        // Kiểm tra đã có relationship accepted chưa
        if (elderSupervisorRepository.existsAcceptedRelationship(elderUser.getId(), supervisorUser.getId())) {
            throw new IllegalArgumentException("Đã có kết nối giữa hai người dùng này");
        }

        // Kiểm tra đã có pending request chưa
        if (elderSupervisorRepository.existsPendingRequest(elderUser.getId(), supervisorUser.getId())) {
            throw new IllegalArgumentException("Đã có yêu cầu đang chờ xử lý");
        }

        // Tạo ElderSupervisor với status PENDING
        ElderSupervisor relationship = new ElderSupervisor()
                .setElderUser(elderUser)
                .setSupervisorUser(supervisorUser)
                .setRequester(requester)
                .setStatus(ERelationshipRequestStatus.PENDING)
                .setRequestMessage(requestDTO.getMessage())
                .setIsActive(false) // Chưa active cho đến khi accept
                .setCanViewPrescription(true) // Mặc định khi accept
                .setCanUpdatePrescription(false); // Mặc định

        ElderSupervisor savedRelationship = elderSupervisorRepository.save(relationship);

        // Tạo notification cho người nhận
        Notifications notification = new Notifications()
                .setUser(target)
                .setNotificationType(ENotificationType.RELATIONSHIP_REQUEST)
                .setTitle("Yêu cầu kết nối mới")
                .setBody(String.format("%s đã gửi yêu cầu kết nối: %s", 
                    requester.getFullName(), requestDTO.getMessage()))
                .setActionUrl("/relationships/pending")
                .setIsRead(false);

        notificationRepository.save(notification);

        log.info("Request sent from user {} to user {}", requesterId, requestDTO.getTargetUserId());

        return mapToResponse(savedRelationship);
    }

    /**
     * Chấp nhận request
     */
    @Transactional
    public RelationshipRequestResponse acceptRequest(Long userId, Long requestId, RespondToRequestDTO responseDTO) {
        ElderSupervisor relationship = elderSupervisorRepository.findById(requestId)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy yêu cầu"));

        // Kiểm tra user là người nhận (không phải requester)
        Long receiverId = relationship.getRequester().getId().equals(relationship.getElderUser().getId()) 
            ? relationship.getSupervisorUser().getId() 
            : relationship.getElderUser().getId();

        if (!userId.equals(receiverId)) {
            throw new IllegalArgumentException("Bạn không có quyền chấp nhận yêu cầu này");
        }

        if (relationship.getStatus() != ERelationshipRequestStatus.PENDING) {
            throw new IllegalArgumentException("Yêu cầu này đã được xử lý");
        }

        // Cập nhật status
        relationship.setStatus(ERelationshipRequestStatus.ACCEPTED)
                    .setRespondedAt(Instant.now())
                    .setResponseMessage(responseDTO.getMessage())
                    .setIsActive(true) // Active permissions
                    .setUpdatedAt(Instant.now());

        ElderSupervisor updatedRelationship = elderSupervisorRepository.save(relationship);

        // Tạo notification cho requester
        User receiver = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng"));

        Notifications notification = new Notifications()
                .setUser(relationship.getRequester())
                .setNotificationType(ENotificationType.RELATIONSHIP_ACCEPTED)
                .setTitle("Yêu cầu kết nối được chấp nhận")
                .setBody(String.format("%s đã chấp nhận yêu cầu kết nối của bạn: %s", 
                    receiver.getFullName(), responseDTO.getMessage()))
                .setActionUrl("/relationships/list")
                .setIsRead(false);

        notificationRepository.save(notification);

        log.info("Request {} accepted by user {}", requestId, userId);

        return mapToResponse(updatedRelationship);
    }

    /**
     * Từ chối request
     */
    @Transactional
    public RelationshipRequestResponse rejectRequest(Long userId, Long requestId, RespondToRequestDTO responseDTO) {
        ElderSupervisor relationship = elderSupervisorRepository.findById(requestId)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy yêu cầu"));

        // Kiểm tra user là người nhận
        Long receiverId = relationship.getRequester().getId().equals(relationship.getElderUser().getId()) 
            ? relationship.getSupervisorUser().getId() 
            : relationship.getElderUser().getId();

        if (!userId.equals(receiverId)) {
            throw new IllegalArgumentException("Bạn không có quyền từ chối yêu cầu này");
        }

        if (relationship.getStatus() != ERelationshipRequestStatus.PENDING) {
            throw new IllegalArgumentException("Yêu cầu này đã được xử lý");
        }

        // Cập nhật status
        relationship.setStatus(ERelationshipRequestStatus.REJECTED)
                    .setRespondedAt(Instant.now())
                    .setResponseMessage(responseDTO.getMessage())
                    .setUpdatedAt(Instant.now());

        ElderSupervisor updatedRelationship = elderSupervisorRepository.save(relationship);

        log.info("Request {} rejected by user {}", requestId, userId);

        // Không gửi notification khi reject (optional)
        return mapToResponse(updatedRelationship);
    }

    /**
     * Lấy danh sách pending requests mà user nhận được
     */
    public List<RelationshipRequestResponse> getPendingReceivedRequests(Long userId) {
        List<ElderSupervisor> requests = elderSupervisorRepository.findPendingReceivedRequests(userId);
        return requests.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    /**
     * Lấy danh sách requests mà user đã gửi
     */
    public List<RelationshipRequestResponse> getSentRequests(Long userId) {
        List<ElderSupervisor> requests = elderSupervisorRepository.findSentRequestsByUserId(userId);
        return requests.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    /**
     * Lấy danh sách pending requests mà user đã gửi
     */
    public List<RelationshipRequestResponse> getPendingSentRequests(Long userId) {
        List<ElderSupervisor> requests = elderSupervisorRepository.findPendingSentRequests(userId);
        return requests.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    /**
     * Lấy danh sách Elder đã kết nối (cho Supervisor)
     */
    public List<RelationshipRequestResponse> getConnectedElders(Long supervisorId) {
        // Kiểm tra user có phải Supervisor không
        User user = userRepository.findById(supervisorId)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng"));
        
        if (!(user instanceof SupervisorUser)) {
            throw new IllegalArgumentException("Chỉ Supervisor mới có thể xem danh sách Elder");
        }
        
        List<ElderSupervisor> relationships = elderSupervisorRepository.findActiveBySupervisorUserId(supervisorId);
        return relationships.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    /**
     * Lấy danh sách Supervisor đã kết nối (cho Elder)
     */
    public List<RelationshipRequestResponse> getConnectedSupervisors(Long elderId) {
        // Kiểm tra user có phải Elder không
        User user = userRepository.findById(elderId)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng"));
        
        if (!(user instanceof ElderUser)) {
            throw new IllegalArgumentException("Chỉ Elder mới có thể xem danh sách Supervisor");
        }
        
        List<ElderSupervisor> relationships = elderSupervisorRepository.findActiveByElderUserId(elderId);
        return relationships.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    private RelationshipRequestResponse mapToResponse(ElderSupervisor relationship) {
        ElderUser elder = relationship.getElderUser();
        SupervisorUser supervisor = relationship.getSupervisorUser();
        
        return RelationshipRequestResponse.builder()
                .id(relationship.getId())
                // Elder info
                .elderUserId(elder.getId())
                .elderUserName(elder.getFullName())
                .elderUserEmail(elder.getEmail())
                .elderUserAvatar(elder.getAvatar())
                .elderUserPhone(elder.getPhoneNumber())
                .elderBloodType(elder.getBloodType())
                .elderHeight(elder.getHeight())
                .elderWeight(elder.getWeight())
                .elderGender(elder.getGender())
                // Supervisor info
                .supervisorUserId(supervisor.getId())
                .supervisorUserName(supervisor.getFullName())
                .supervisorUserEmail(supervisor.getEmail())
                .supervisorUserAvatar(supervisor.getAvatar())
                .supervisorOccupation(supervisor.getOccupation())
                .supervisorWorkplace(supervisor.getWorkplace())
                // Request info
                .requesterId(relationship.getRequester().getId())
                .requesterName(relationship.getRequester().getFullName())
                .status(relationship.getStatus())
                .requestMessage(relationship.getRequestMessage())
                .responseMessage(relationship.getResponseMessage())
                // Permissions
                .canViewMedications(relationship.getCanViewPrescription())
                .canUpdateMedications(relationship.getCanUpdatePrescription())
                .respondedAt(relationship.getRespondedAt())
                .createdAt(relationship.getCreatedAt())
                .build();
    }

    // ==================== EMAIL-BASED METHODS ====================
    
    /**
     * Gửi request kết nối - nhận email thay vì userId
     */
    @Transactional
    public RelationshipRequestResponse sendRequestByEmail(String email, SendRelationshipRequestDTO requestDTO) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return sendRequest(user.getId(), requestDTO);
    }

    /**
     * Chấp nhận yêu cầu kết nối - nhận email
     */
    @Transactional
    public RelationshipRequestResponse acceptRequestByEmail(String email, Long requestId, RespondToRequestDTO responseDTO) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return acceptRequest(user.getId(), requestId, responseDTO);
    }

    /**
     * Từ chối yêu cầu kết nối - nhận email
     */
    @Transactional
    public RelationshipRequestResponse rejectRequestByEmail(String email, Long requestId, RespondToRequestDTO responseDTO) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return rejectRequest(user.getId(), requestId, responseDTO);
    }

    /**
     * Lấy danh sách pending requests đã nhận - nhận email
     */
    public List<RelationshipRequestResponse> getPendingReceivedRequestsByEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return getPendingReceivedRequests(user.getId());
    }

    /**
     * Lấy danh sách requests đã gửi - nhận email
     */
    public List<RelationshipRequestResponse> getSentRequestsByEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return getSentRequests(user.getId());
    }

    /**
     * Lấy danh sách pending requests đã gửi - nhận email
     */
    public List<RelationshipRequestResponse> getPendingSentRequestsByEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return getPendingSentRequests(user.getId());
    }

    /**
     * Lấy danh sách Elder đã kết nối (cho Supervisor) - nhận email
     */
    public List<RelationshipRequestResponse> getConnectedEldersByEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return getConnectedElders(user.getId());
    }

    /**
     * Lấy danh sách Supervisor đã kết nối (cho Elder) - nhận email
     */
    public List<RelationshipRequestResponse> getConnectedSupervisorsByEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return getConnectedSupervisors(user.getId());
    }

    /**
     * Lấy quyền của Supervisor đối với Elder (getRole)
     * @param supervisorId ID của Supervisor
     * @param elderId ID của Elder
     * @return SupervisorPermissionResponse chứa thông tin quyền
     */
    public SupervisorPermissionResponse getRole(Long supervisorId, Long elderId) {
        log.info("Getting permissions for supervisor {} and elder {}", supervisorId, elderId);
        
        // Tìm Supervisor
        SupervisorUser supervisor = userRepository.findById(supervisorId)
                .filter(user -> user instanceof SupervisorUser)
                .map(user -> (SupervisorUser) user)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy Supervisor với ID: " + supervisorId));
        
        // Tìm Elder
        ElderUser elder = userRepository.findById(elderId)
                .filter(user -> user instanceof ElderUser)
                .map(user -> (ElderUser) user)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy Elder với ID: " + elderId));
        
        // Tìm relationship
        ElderSupervisor relationship = elderSupervisorRepository
                .findByElderUserIdAndSupervisorUserIdAndStatus(
                    elderId, 
                    supervisorId, 
                    ERelationshipRequestStatus.ACCEPTED
                )
                .orElseThrow(() -> new ResourceNotFoundException(
                    "Không tìm thấy mối quan hệ giữa Supervisor và Elder này"
                ));
        
        log.info("Found relationship - canView: {}, canUpdate: {}, isActive: {}", 
            relationship.getCanViewPrescription(), 
            relationship.getCanUpdatePrescription(),
            relationship.getIsActive());
        
        return SupervisorPermissionResponse.builder()
                .supervisorId(supervisor.getId())
                .supervisorName(supervisor.getFullName())
                .supervisorEmail(supervisor.getEmail())
                .elderId(elder.getId())
                .elderName(elder.getFullName())
                .elderEmail(elder.getEmail())
                .canViewMedications(relationship.getCanViewPrescription())
                .canUpdateMedications(relationship.getCanUpdatePrescription())
                .isActive(relationship.getIsActive())
                .relationshipStatus(relationship.getStatus().name())
                .build();
    }

    /**
     * Helper method: Lấy userId từ email
     */
    public Long getUserIdByEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy người dùng với email: " + email));
        return user.getId();
    }

    /**
     * Cập nhật quyền của Supervisor (chỉ Elder mới được gọi)
     * @param elderEmail Email của Elder đang đăng nhập
     * @param supervisorId ID của Supervisor cần cập nhật quyền
     * @param canViewMedications Quyền xem thuốc
     * @param canUpdateMedications Quyền sửa/xóa thuốc
     * @return SupervisorPermissionResponse với thông tin quyền đã cập nhật
     */
    @Transactional
    public SupervisorPermissionResponse updatePermissions(
            String elderEmail,
            Long supervisorId,
            Boolean canViewMedications,
            Boolean canUpdateMedications
    ) {
        log.info("Updating permissions - Elder: {}, Supervisor: {}, View: {}, Update: {}", 
                elderEmail, supervisorId, canViewMedications, canUpdateMedications);
        
        // Tìm Elder từ email
        ElderUser elder = userRepository.findByEmail(elderEmail)
                .filter(user -> user instanceof ElderUser)
                .map(user -> (ElderUser) user)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy Elder với email: " + elderEmail));
        
        // Tìm Supervisor
        SupervisorUser supervisor = userRepository.findById(supervisorId)
                .filter(user -> user instanceof SupervisorUser)
                .map(user -> (SupervisorUser) user)
                .orElseThrow(() -> new ResourceNotFoundException("Không tìm thấy Supervisor với ID: " + supervisorId));
        
        // Tìm relationship ACCEPTED giữa Elder và Supervisor
        ElderSupervisor relationship = elderSupervisorRepository
                .findByElderUserIdAndSupervisorUserIdAndStatus(
                    elder.getId(),
                    supervisorId,
                    ERelationshipRequestStatus.ACCEPTED
                )
                .orElseThrow(() -> new ResourceNotFoundException(
                    "Không tìm thấy mối quan hệ đã chấp nhận giữa Elder và Supervisor"
                ));
        
        // Cập nhật quyền
        relationship.setCanViewPrescription(canViewMedications);
        relationship.setCanUpdatePrescription(canUpdateMedications);
        
        ElderSupervisor updatedRelationship = elderSupervisorRepository.save(relationship);
        
        log.info("Permissions updated successfully for relationship ID: {}", updatedRelationship.getId());
        
        // Tạo notification cho Supervisor
        createPermissionUpdateNotification(elder, supervisor, canViewMedications, canUpdateMedications);
        
        // Return response
        return SupervisorPermissionResponse.builder()
                .supervisorId(supervisor.getId())
                .supervisorName(supervisor.getFullName())
                .elderId(elder.getId())
                .elderName(elder.getFullName())
                .canViewMedications(canViewMedications)
                .canUpdateMedications(canUpdateMedications)
                .isActive(updatedRelationship.getIsActive())
                .relationshipStatus(updatedRelationship.getStatus().name())
                .build();
    }

    /**
     * Tạo notification khi Elder thay đổi quyền của Supervisor
     */
    private void createPermissionUpdateNotification(
            ElderUser elder,
            SupervisorUser supervisor,
            Boolean canView,
            Boolean canUpdate
    ) {
        String permissionText;
        if (canView && canUpdate) {
            permissionText = "xem và chỉnh sửa thuốc";
        } else if (canView) {
            permissionText = "chỉ xem thuốc (không được chỉnh sửa)";
        } else {
            permissionText = "không được xem thuốc";
        }
        
        Notifications notification = new Notifications();
        notification.setUser(supervisor); // Gửi cho Supervisor
        notification.setNotificationType(ENotificationType.SYSTEM_ANNOUNCEMENT);
        notification.setTitle("Quyền truy cập đã thay đổi");
        notification.setBody(String.format("%s đã cập nhật quyền của bạn: %s", 
                elder.getFullName(), permissionText));
        notification.setIsRead(false);
        notification.setRelatedElder(elder);
        
        notificationRepository.save(notification);
        log.info("Created permission update notification for Supervisor: {}", supervisor.getEmail());
    }
}
