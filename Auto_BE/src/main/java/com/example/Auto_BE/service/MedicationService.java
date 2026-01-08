package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.CreateMedicationRequest;
import com.example.Auto_BE.dto.request.UpdateMedicationRequest;
import com.example.Auto_BE.dto.response.MedicationResponse;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Prescriptions;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.entity.enums.ETypeMedication;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.ElderSupervisorRepository;
import com.example.Auto_BE.repository.ElderUserRepository;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import com.example.Auto_BE.repository.PrescriptionRepository;
import com.example.Auto_BE.repository.SupervisorUserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@Transactional
public class MedicationService {

    @Autowired
    private MedicationReminderRepository medicationReminderRepository;
    
    @Autowired
    private ElderUserRepository elderUserRepository;
    
    @Autowired
    private SupervisorUserRepository supervisorUserRepository;
    
    @Autowired
    private PrescriptionRepository prescriptionRepository;
    
    @Autowired
    private SimpleTimeBasedScheduler simpleTimeBasedScheduler;
    
    @Autowired
    private ElderSupervisorRepository elderSupervisorRepository;

    /**
     * Tạo medication reminder mới và auto schedule
     * Response trả LIST của medications (1 medication per time)
     */
    public BaseResponse<List<MedicationResponse>> createMedication(CreateMedicationRequest request, Authentication authentication) {
        try {
            System.out.println("Creating medication: " + request.getName() + ", ElderUserId from request: " + request.getElderUserId() + ", Auth email: " + authentication.getName());

            // Nếu có elderUserId → Supervisor tạo cho Elder
            ElderUser elderUser;
            if (request.getElderUserId() != null) {
                System.out.println("Supervisor mode - Finding elder by ID: " + request.getElderUserId());
                elderUser = elderUserRepository.findById(request.getElderUserId())
                        .orElseThrow(() -> new BaseException.EntityNotFoundException("Elder user không tồn tại với ID: " + request.getElderUserId()));
                
                System.out.println("Found elder user: " + elderUser.getEmail() + " (ID: " + elderUser.getId() + ")");
                
                // Verify Supervisor permission
                System.out.println("Finding supervisor by email: " + authentication.getName());
                SupervisorUser supervisorUser = supervisorUserRepository.findByEmail(authentication.getName())
                        .orElseThrow(() -> new BaseException.EntityNotFoundException("User không tồn tại: " + authentication.getName()));

                System.out.println("Found supervisor user: " + supervisorUser.getEmail() + " (ID: " + supervisorUser.getId() + ")");
                
                System.out.println("Checking permission: supervisor ID " + supervisorUser.getId() + " for elder ID " + elderUser.getId());
                boolean hasPermission = elderSupervisorRepository
                        .findActiveWithUpdatePermission(supervisorUser.getId(), elderUser.getId())
                        .isPresent();

                System.out.println("Has permission: " + hasPermission);
                
                if (!hasPermission) {
                    throw new BaseException.BadRequestException("Không có quyền tạo medication cho Elder này");
                }
                
                System.out.println("Supervisor " + supervisorUser.getId() + " creating medication for Elder " + elderUser.getId());
            } else {
                System.out.println("Elder mode - Finding elder by email: " + authentication.getName());
                // Elder tự tạo cho mình
                elderUser = elderUserRepository.findByEmail(authentication.getName())
                        .orElseThrow(() -> new RuntimeException("Elder user không tồn tại: " + authentication.getName()));
                System.out.println("Found elder user: " + elderUser.getEmail() + " (ID: " + elderUser.getId() + ")");
            }

            // Validate prescription exists (optional)
            Prescriptions prescription = null;
            if (request.getPrescriptionId() != null) {
                prescription = prescriptionRepository.findById(request.getPrescriptionId())
                        .orElseThrow(() -> new RuntimeException("Prescription không tồn tại"));
            }

            // Tạo NHIỀU MedicationReminders (1 per time)
            List<MedicationReminder> savedMedications = new java.util.ArrayList<>();
            
            for (String reminderTime : request.getReminderTimes()) {
                MedicationReminder medication = new MedicationReminder();
                medication.setName(request.getName());
                medication.setDescription(request.getDescription());
                medication.setType(request.getType());
                medication.setReminderTime(reminderTime);
                medication.setDaysOfWeek(request.getDaysOfWeek());
                medication.setIsActive(request.getIsActive());
                medication.setElderUser(elderUser);
                medication.setPrescription(prescription);

                MedicationReminder saved = medicationReminderRepository.save(medication);
                savedMedications.add(saved);
                System.out.println("Medication created with ID: " + saved.getId() + " at " + reminderTime);
            }

            // Auto schedule TIME-BASED reminders cho Elder
            try {
                simpleTimeBasedScheduler.scheduleUserReminders(elderUser.getId());
                System.out.println("Auto-scheduled TIME-BASED reminders for elder user: " + elderUser.getId());
            } catch (Exception e) {
                System.err.println("Medication created but scheduling failed: " + e.getMessage());
            }

            // Response trả LIST - mỗi medication 1 entry
            List<MedicationResponse> responseList = savedMedications.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<MedicationResponse>>builder()
                    .status("success")
                    .message("Tạo thành công " + savedMedications.size() + " medications")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            System.err.println("Error creating medication: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<List<MedicationResponse>>builder()
                    .status("error")
                    .message("Lỗi khi tạo medication: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy medication theo ID
     */
    public BaseResponse<MedicationResponse> getMedicationById(Long medicationId, Authentication authentication) {
        try {
            Optional<MedicationReminder> medicationOpt = medicationReminderRepository.findById(medicationId);
            
            if (!medicationOpt.isPresent()) {
                return BaseResponse.<MedicationResponse>builder()
                        .status("error")
                        .message("Medication không tồn tại")
                        .data(null)
                        .build();
            }

            MedicationReminder medication = medicationOpt.get();
            MedicationResponse response = convertToResponse(medication);

            return BaseResponse.<MedicationResponse>builder()
                    .status("success")
                    .message("Lấy medication thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            return BaseResponse.<MedicationResponse>builder()
                    .status("error")
                    .message("Lỗi khi lấy medication: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy tất cả medications của user (cả prescription và standalone)
     */
    public BaseResponse<List<MedicationResponse>> getAllMedicationsByUser(Long userId) {
        try {
            List<MedicationReminder> medications = medicationReminderRepository
                    .findAll()
                    .stream()
                    .filter(med -> med.getElderUser().getId().equals(userId))
                    .collect(Collectors.toList());

            List<MedicationResponse> responseList = medications.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<MedicationResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách medications thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            return BaseResponse.<List<MedicationResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách medications: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy chỉ standalone medications (thuốc ngoài đơn)
     * prescriptionId = null HOẶC type = OVER_THE_COUNTER
     * Group các medications trùng HOÀN TOÀN (name, daysOfWeek, description, type, isActive)
     * và gộp reminderTimes
     */
    public BaseResponse<List<MedicationResponse>> getStandaloneMedicationsByUser(Long userId) {
        try {
            // Lọc medications: prescriptionId = null HOẶC type = OVER_THE_COUNTER
            List<MedicationReminder> standaloneMedications = medicationReminderRepository
                    .findAll()
                    .stream()
                    .filter(med -> med.getElderUser().getId().equals(userId))
                    .filter(med -> med.getPrescription() == null || 
                                   med.getType() == ETypeMedication.OVER_THE_COUNTER)
                    .collect(Collectors.toList());

            // Group theo composite key: name + daysOfWeek + description + type + isActive
            Map<String, List<MedicationReminder>> groupedMedications = standaloneMedications.stream()
                    .collect(Collectors.groupingBy(med -> {
                        // Tạo composite key để group
                        String name = med.getName();
                        String daysOfWeek = med.getDaysOfWeek();
                        String description = med.getDescription() != null ? med.getDescription() : "";
                        String type = med.getType().toString();
                        String isActive = med.getIsActive().toString();
                        
                        return name + "|" + daysOfWeek + "|" + description + "|" + type + "|" + isActive;
                    }));

            List<MedicationResponse> responseList = groupedMedications.entrySet().stream()
                    .map(entry -> {
                        List<MedicationReminder> meds = entry.getValue();
                        
                        // Lấy medication đầu tiên làm template
                        MedicationReminder first = meds.get(0);
                        
                        // Gộp tất cả reminderTimes và sort
                        List<String> allReminderTimes = meds.stream()
                                .map(MedicationReminder::getReminderTime)
                                .sorted() // Sort để hiển thị theo thứ tự thời gian
                                .collect(Collectors.toList());
                        
                        // Build response với reminderTimes đã gộp
                        return MedicationResponse.builder()
                                .id(first.getId())
                                .userId(first.getElderUser().getId())
                                .userName(first.getElderUser().getFullName())
                                .prescriptionId(first.getPrescription() != null ? first.getPrescription().getId() : null)
                                .medicationName(first.getName())
                                .type(first.getType())
                                .reminderTimes(allReminderTimes)
                                .daysOfWeek(first.getDaysOfWeek())
                                .description(first.getDescription())
                                .isActive(first.getIsActive())
                                .createdAt(first.getCreatedAt() != null ? 
                                    java.time.LocalDateTime.ofInstant(first.getCreatedAt(), java.time.ZoneId.systemDefault()) : null)
                                .updatedAt(first.getUpdatedAt() != null ? 
                                    java.time.LocalDateTime.ofInstant(first.getUpdatedAt(), java.time.ZoneId.systemDefault()) : null)
                                .build();
                    })
                    .collect(Collectors.toList());

            return BaseResponse.<List<MedicationResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách standalone medications thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            return BaseResponse.<List<MedicationResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách standalone medications: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Update medication và auto reschedule
     */
    public BaseResponse<MedicationResponse> updateMedication(Long medicationId, UpdateMedicationRequest request, Authentication authentication) {
        try {
            System.out.println("Updating medication ID: " + medicationId + ", Auth email: " + authentication.getName());

            MedicationReminder medication = medicationReminderRepository.findById(medicationId)
                    .orElseThrow(() -> new RuntimeException("Medication không tồn tại"));

            System.out.println("Found medication owned by elder ID: " + medication.getElderUser().getId());

            // Verify permission (Elder owner or Supervisor with permission)
            ElderUser authUserAsElder = elderUserRepository.findByEmail(authentication.getName()).orElse(null);
            SupervisorUser authUserAsSupervisor = supervisorUserRepository.findByEmail(authentication.getName()).orElse(null);

            boolean isOwner = false;
            boolean isSupervisorWithPermission = false;

            if (authUserAsElder != null) {
                isOwner = medication.getElderUser().getId().equals(authUserAsElder.getId());
                System.out.println("Is owner (Elder): " + isOwner);
            }

            if (authUserAsSupervisor != null) {
                isSupervisorWithPermission = elderSupervisorRepository
                        .findActiveWithUpdatePermission(authUserAsSupervisor.getId(), medication.getElderUser().getId())
                        .isPresent();
                System.out.println("Is supervisor with permission: " + isSupervisorWithPermission);
            }

            if (!isOwner && !isSupervisorWithPermission) {
                return BaseResponse.<MedicationResponse>builder()
                        .status("error")
                        .message("Không có quyền update medication này")
                        .data(null)
                        .build();
            }

            // Update fields (chỉ update fields được gửi)
            if (request.getName() != null) {
                medication.setName(request.getName());
            }
            if (request.getDescription() != null) {
                medication.setDescription(request.getDescription());
            }
            if (request.getType() != null) {
                medication.setType(request.getType());
            }
            if (request.getDaysOfWeek() != null) {
                medication.setDaysOfWeek(request.getDaysOfWeek());
            }
            if (request.getIsActive() != null) {
                medication.setIsActive(request.getIsActive());
            }
            
            // Xử lý reminderTimes array
            if (request.getReminderTimes() != null && !request.getReminderTimes().isEmpty()) {
                // Nếu update reminderTimes, cần xóa medication cũ và tạo mới
                // hoặc chỉ update reminderTimeSimple của medication hiện tại
                String newReminderTime = request.getReminderTimes().get(0);
                medication.setReminderTime(newReminderTime);
                System.out.println("Updated reminder time to: " + newReminderTime);
            }

            // Save
            MedicationReminder updatedMedication = medicationReminderRepository.save(medication);
            System.out.println("Medication updated: " + updatedMedication.getId());

            // Auto reschedule TIME-BASED reminders
            Long elderUserId = medication.getElderUser().getId();
            try {
                simpleTimeBasedScheduler.scheduleUserReminders(elderUserId);
                System.out.println("Auto-rescheduled TIME-BASED reminders for user: " + elderUserId);
            } catch (Exception e) {
                System.err.println("Medication updated but rescheduling failed: " + e.getMessage());
            }

            // Convert to response
            MedicationResponse response = convertToResponse(updatedMedication);

            return BaseResponse.<MedicationResponse>builder()
                    .status("success")
                    .message("Update medication thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            System.err.println("Error updating medication: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<MedicationResponse>builder()
                    .status("error")
                    .message("Lỗi khi update medication: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Xóa medication và auto reschedule
     */
    public BaseResponse<String> deleteMedication(Long medicationId, Authentication authentication) {
        try {
            System.out.println("Deleting medication ID: " + medicationId + ", Auth email: " + authentication.getName());

            MedicationReminder medication = medicationReminderRepository.findById(medicationId)
                    .orElseThrow(() -> new RuntimeException("Medication không tồn tại"));

            System.out.println("Found medication owned by elder ID: " + medication.getElderUser().getId());

            // Verify permission (Elder owner or Supervisor with permission)
            ElderUser authUserAsElder = elderUserRepository.findByEmail(authentication.getName()).orElse(null);
            SupervisorUser authUserAsSupervisor = supervisorUserRepository.findByEmail(authentication.getName()).orElse(null);

            boolean isOwner = false;
            boolean isSupervisorWithPermission = false;
            Long elderUserId = medication.getElderUser().getId();

            if (authUserAsElder != null) {
                isOwner = elderUserId.equals(authUserAsElder.getId());
                System.out.println("Is owner (Elder): " + isOwner);
            }

            if (authUserAsSupervisor != null) {
                isSupervisorWithPermission = elderSupervisorRepository
                        .findActiveWithUpdatePermission(authUserAsSupervisor.getId(), elderUserId)
                        .isPresent();
                System.out.println("Is supervisor with permission: " + isSupervisorWithPermission);
            }

            if (!isOwner && !isSupervisorWithPermission) {
                return BaseResponse.<String>builder()
                        .status("error")
                        .message("Không có quyền xóa medication này")
                        .data(null)
                        .build();
            }

            // Delete
            medicationReminderRepository.delete(medication);
            System.out.println("Medication deleted: " + medicationId);

            // Auto reschedule TIME-BASED reminders (cho medications còn lại)
            try {
                simpleTimeBasedScheduler.scheduleUserReminders(elderUserId);
                System.out.println("Auto-rescheduled remaining TIME-BASED reminders for user: " + elderUserId);
            } catch (Exception e) {
                System.err.println("Medication deleted but rescheduling failed: " + e.getMessage());
            }

            return BaseResponse.<String>builder()
                    .status("success")
                    .message("Xóa medication thành công")
                    .data("Medication ID " + medicationId + " đã được xóa")
                    .build();

        } catch (Exception e) {
            System.err.println("Error deleting medication: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<String>builder()
                    .status("error")
                    .message("Lỗi khi xóa medication: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Toggle trạng thái active/inactive của medication
     */
    public BaseResponse<MedicationResponse> toggleMedicationStatus(Long medicationId, Authentication authentication) {
        try {
            System.out.println("Toggling medication status ID: " + medicationId + ", Auth email: " + authentication.getName());

            // Find medication
            MedicationReminder medication = medicationReminderRepository.findById(medicationId)
                    .orElseThrow(() -> new RuntimeException("Medication không tồn tại"));

            System.out.println("Found medication owned by elder ID: " + medication.getElderUser().getId());

            // Check authorization (Elder owner or Supervisor with permission)
            ElderUser authUserAsElder = elderUserRepository.findByEmail(authentication.getName()).orElse(null);
            SupervisorUser authUserAsSupervisor = supervisorUserRepository.findByEmail(authentication.getName()).orElse(null);

            boolean isOwner = false;
            boolean isSupervisorWithPermission = false;
            Long elderUserId = medication.getElderUser().getId();

            if (authUserAsElder != null) {
                isOwner = elderUserId.equals(authUserAsElder.getId());
                System.out.println("Is owner (Elder): " + isOwner);
            }

            if (authUserAsSupervisor != null) {
                isSupervisorWithPermission = elderSupervisorRepository
                        .findActiveWithUpdatePermission(authUserAsSupervisor.getId(), elderUserId)
                        .isPresent();
                System.out.println("Is supervisor with permission: " + isSupervisorWithPermission);
            }

            if (!isOwner && !isSupervisorWithPermission) {
                throw new RuntimeException("Không có quyền thao tác medication này");
            }

            // Toggle status
            medication.setIsActive(!medication.getIsActive());
            MedicationReminder updatedMedication = medicationReminderRepository.save(medication);

            System.out.println("Medication status toggled to: " + updatedMedication.getIsActive());

            // Auto reschedule reminders
            try {
                simpleTimeBasedScheduler.scheduleUserReminders(elderUserId);
                System.out.println("Auto-rescheduled TIME-BASED reminders for user: " + elderUserId);
            } catch (Exception e) {
                System.err.println("Status toggled but rescheduling failed: " + e.getMessage());
            }

            // Convert to response
            MedicationResponse response = convertToResponse(updatedMedication);

            return BaseResponse.<MedicationResponse>builder()
                    .status("success")
                    .message("Toggle medication status thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            System.err.println("Error toggling medication status: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<MedicationResponse>builder()
                    .status("error")
                    .message("Lỗi khi toggle medication status: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    // ===== HELPER METHODS =====

//    private ElderUser getUserById(Long userId) {
//        Optional<ElderUser> userOpt = elderUserRepository.findById(userId);
//        return userOpt.orElse(null);
//    }

    /**
     * Không gom medications, chỉ return đúng medication được tạo
     */
    private MedicationResponse convertToResponse(MedicationReminder medication) {
        // KHÔNG GOM NỮA - Chỉ return medication hiện tại
        // Vì createMedication() đã tạo nhiều records, response chỉ cần show 1 record
        
        return MedicationResponse.builder()
                .id(medication.getId())
                .userId(medication.getElderUser().getId())
                .userName(medication.getElderUser().getFullName())
                .prescriptionId(medication.getPrescription() != null ? medication.getPrescription().getId() : null)
                .medicationName(medication.getName())
                .type(medication.getType())
                .reminderTimes(java.util.Arrays.asList(medication.getReminderTime()))  // Chỉ time của medication này
                .daysOfWeek(medication.getDaysOfWeek())
                .description(medication.getDescription())
                .isActive(medication.getIsActive())
                .createdAt(medication.getCreatedAt() != null ? 
                    java.time.LocalDateTime.ofInstant(medication.getCreatedAt(), java.time.ZoneId.systemDefault()) : null)
                .updatedAt(medication.getUpdatedAt() != null ? 
                    java.time.LocalDateTime.ofInstant(medication.getUpdatedAt(), java.time.ZoneId.systemDefault()) : null)
                .build();
    }
}