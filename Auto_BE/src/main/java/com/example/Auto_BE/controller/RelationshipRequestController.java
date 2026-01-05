package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.request.RespondToRequestDTO;
import com.example.Auto_BE.dto.request.SendRelationshipRequestDTO;
import com.example.Auto_BE.dto.response.RelationshipRequestResponse;
import com.example.Auto_BE.dto.response.SupervisorPermissionResponse;
import com.example.Auto_BE.service.RelationshipRequestService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;

@RestController
@RequestMapping("/api/relationships")
@RequiredArgsConstructor
public class RelationshipRequestController {

    private final RelationshipRequestService relationshipRequestService;

    /**
     * Gửi yêu cầu kết nối
     * POST /api/relationships/request
     */
    @PostMapping("/request")
    public ResponseEntity<RelationshipRequestResponse> sendRequest(
            Authentication authentication,
            @Valid @RequestBody SendRelationshipRequestDTO requestDTO) {
        
        String email = authentication.getName();
        RelationshipRequestResponse response = relationshipRequestService.sendRequestByEmail(email, requestDTO);
        
        return ResponseEntity.ok(response);
    }

    /**
     * Chấp nhận yêu cầu kết nối
     * PUT /api/relationships/{id}/accept
     */
    @PutMapping("/{id}/accept")
    public ResponseEntity<RelationshipRequestResponse> acceptRequest(
            Authentication authentication,
            @PathVariable Long id,
            @RequestBody RespondToRequestDTO responseDTO) {
        
        String email = authentication.getName();
        RelationshipRequestResponse response = relationshipRequestService.acceptRequestByEmail(email, id, responseDTO);
        
        return ResponseEntity.ok(response);
    }

    /**
     * Từ chối yêu cầu kết nối
     * PUT /api/relationships/{id}/reject
     */
    @PutMapping("/{id}/reject")
    public ResponseEntity<RelationshipRequestResponse> rejectRequest(
            Authentication authentication,
            @PathVariable Long id,
            @RequestBody RespondToRequestDTO responseDTO) {
        
        String email = authentication.getName();
        RelationshipRequestResponse response = relationshipRequestService.rejectRequestByEmail(email, id, responseDTO);
        
        return ResponseEntity.ok(response);
    }

    /**
     * Lấy danh sách yêu cầu pending mà user nhận được
     * GET /api/relationships/pending/received
     */
    @GetMapping("/pending/received")
    public ResponseEntity<List<RelationshipRequestResponse>> getPendingReceivedRequests(
            Authentication authentication) {
        
        String email = authentication.getName();
        List<RelationshipRequestResponse> requests = relationshipRequestService.getPendingReceivedRequestsByEmail(email);
        
        return ResponseEntity.ok(requests);
    }

    /**
     * Lấy danh sách yêu cầu mà user đã gửi
     * GET /api/relationships/sent
     */
    @GetMapping("/sent")
    public ResponseEntity<List<RelationshipRequestResponse>> getSentRequests(
            Authentication authentication) {
        
        String email = authentication.getName();
        List<RelationshipRequestResponse> requests = relationshipRequestService.getSentRequestsByEmail(email);
        
        return ResponseEntity.ok(requests);
    }

    /**
     * Lấy danh sách yêu cầu pending mà user đã gửi
     * GET /api/relationships/pending/sent
     */
    @GetMapping("/pending/sent")
    public ResponseEntity<List<RelationshipRequestResponse>> getPendingSentRequests(
            Authentication authentication) {
        
        String email = authentication.getName();
        List<RelationshipRequestResponse> requests = relationshipRequestService.getPendingSentRequestsByEmail(email);
        
        return ResponseEntity.ok(requests);
    }

    /**
     * Lấy danh sách Elder đã kết nối (cho Supervisor)
     * GET /api/relationships/elders
     * Service sẽ tự kiểm tra user có phải Supervisor không
     */
    @GetMapping("/elders")
    public ResponseEntity<List<RelationshipRequestResponse>> getConnectedElders(
            Authentication authentication) {
        
        String email = authentication.getName();
        List<RelationshipRequestResponse> elders = relationshipRequestService.getConnectedEldersByEmail(email);
        
        return ResponseEntity.ok(elders);
    }

    /**
     * Lấy danh sách Supervisor đã kết nối (cho Elder)
     * GET /api/relationships/supervisors
     * Service sẽ tự kiểm tra user có phải Elder không
     */
    @GetMapping("/supervisors")
    public ResponseEntity<List<RelationshipRequestResponse>> getConnectedSupervisors(
            Authentication authentication) {
        
        String email = authentication.getName();
        List<RelationshipRequestResponse> supervisors = relationshipRequestService.getConnectedSupervisorsByEmail(email);
        
        return ResponseEntity.ok(supervisors);
    }

    /**
     * Lấy quyền của Supervisor đối với Elder (getRole)
     * GET /api/relationships/role/{elderId}
     * 
     * @param elderId ID của Elder cần kiểm tra quyền
     * @param authentication Thông tin Supervisor đang đăng nhập
     * @return SupervisorPermissionResponse chứa thông tin quyền (canViewMedications, canUpdateMedications)
     */
    @GetMapping("/role/{elderId}")
    public ResponseEntity<SupervisorPermissionResponse> getRole(
            @PathVariable Long elderId,
            Authentication authentication) {
        
        String email = authentication.getName();
        
        // Lấy supervisorId từ email
        Long supervisorId = relationshipRequestService.getUserIdByEmail(email);
        
        SupervisorPermissionResponse permissions = relationshipRequestService.getRole(supervisorId, elderId);
        
        return ResponseEntity.ok(permissions);
    }

    /**
     * Cập nhật quyền của Supervisor (chỉ Elder mới được gọi)
     * PUT /api/relationships/permissions/{supervisorId}
     * 
     * @param supervisorId ID của Supervisor cần cập nhật quyền
     * @param canViewMedications Quyền xem thuốc (true/false)
     * @param canUpdateMedications Quyền sửa/xóa thuốc (true/false)
     * @param authentication Thông tin Elder đang đăng nhập
     * @return SupervisorPermissionResponse với thông tin quyền đã cập nhật
     */
    @PutMapping("/permissions/{supervisorId}")
    public ResponseEntity<SupervisorPermissionResponse> updatePermissions(
            @PathVariable Long supervisorId,
            @RequestParam Boolean canViewMedications,
            @RequestParam Boolean canUpdateMedications,
            Authentication authentication) {
        
        String elderEmail = authentication.getName();
        
        SupervisorPermissionResponse updatedPermissions = relationshipRequestService.updatePermissions(
            elderEmail, 
            supervisorId, 
            canViewMedications, 
            canUpdateMedications
        );
        
        return ResponseEntity.ok(updatedPermissions);
    }
}
