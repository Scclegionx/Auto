package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.MedicationReminderResponse;
import com.example.Auto_BE.dto.response.MedicationResponse;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.entity.*;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.mapper.PrescriptionMapper;
import com.example.Auto_BE.repository.*;
import jakarta.transaction.Transactional;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

import static com.example.Auto_BE.constants.ErrorMessages.*;
import static com.example.Auto_BE.constants.SuccessMessage.*;

@Service
public class CronPrescriptionService {
    private final ElderUserRepository elderUserRepository;
    private final SupervisorUserRepository supervisorUserRepository;
    private final UserRepository userRepository;
    private final PrescriptionRepository prescriptionRepository;
    private final MedicationReminderRepository medicationReminderRepository;
    private final SimpleTimeBasedScheduler simpleTimeBasedScheduler; // NEW - TIME-BASED SCHEDULING
    private final ElderSupervisorRepository elderSupervisorRepository;

    public CronPrescriptionService(ElderUserRepository elderUserRepository, 
                                   SupervisorUserRepository supervisorUserRepository,
                                   UserRepository userRepository,
                                   PrescriptionRepository prescriptionRepository,
                                   MedicationReminderRepository medicationReminderRepository,
                                   SimpleTimeBasedScheduler simpleTimeBasedScheduler,
                                   ElderSupervisorRepository elderSupervisorRepository) {
        this.elderUserRepository = elderUserRepository;
        this.supervisorUserRepository = supervisorUserRepository;
        this.userRepository = userRepository;
        this.prescriptionRepository = prescriptionRepository;
        this.medicationReminderRepository = medicationReminderRepository;
        this.simpleTimeBasedScheduler = simpleTimeBasedScheduler;
                this.elderSupervisorRepository = elderSupervisorRepository;
    }

    @Transactional
    public BaseResponse<PrescriptionResponse> create(PrescriptionCreateRequest prescriptionCreateRequest,
                                                     Authentication authentication) {
        System.out.println("Creating prescription. ElderUserId from request: " + 
                prescriptionCreateRequest.getElderUserId() + ", Auth email: " + authentication.getName());
        
        // Nếu có elderUserId → Supervisor tạo cho Elder
        ElderUser elderUser;
        if (prescriptionCreateRequest.getElderUserId() != null) {
            elderUser = elderUserRepository.findById(prescriptionCreateRequest.getElderUserId())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException("Elder user không tồn tại với ID: " + prescriptionCreateRequest.getElderUserId()));
            
            System.out.println("Found elder user: " + elderUser.getEmail() + " (ID: " + elderUser.getId() + ")");
            
            // Verify Supervisor permission
            System.out.println("Finding supervisor by email: " + authentication.getName());
            SupervisorUser supervisorUser = supervisorUserRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND + ": " + authentication.getName()));

            System.out.println("Found supervisor user: " + supervisorUser.getEmail() + " (ID: " + supervisorUser.getId() + ")");
            
            System.out.println("Checking permission: supervisor ID " + supervisorUser.getId() + " for elder ID " + elderUser.getId());
            boolean hasPermission = elderSupervisorRepository
                    .findActiveWithUpdatePermission(supervisorUser.getId(), elderUser.getId())
                    .isPresent();

            if (!hasPermission) {
                throw new BaseException.BadRequestException(PERMISSION_ERROR);
            }
        } else {
            System.out.println("Elder mode - Finding elder by email: " + authentication.getName());
            // Elder tự tạo cho mình
            elderUser = elderUserRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND + ": " + authentication.getName()));
            System.out.println("Found elder user: " + elderUser.getEmail() + " (ID: " + elderUser.getId() + ")");
        }

//        // 2. Validate medication reminders data
//        validateMedicationReminders(prescriptionCreateRequest.getMedicationReminders());

        Prescriptions prescription = PrescriptionMapper.toEntity(prescriptionCreateRequest, elderUser);

        if (prescriptionCreateRequest.getMedicationReminders() != null &&
                !prescriptionCreateRequest.getMedicationReminders().isEmpty()) {

            List<MedicationReminder> medicationReminders = prescriptionCreateRequest.getMedicationReminders().stream()
                    .map(mr -> PrescriptionMapper.toEntities(mr, elderUser, prescription))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());

            prescription.setMedicationReminders(medicationReminders);
        }

        Prescriptions savedPrescription = prescriptionRepository.save(prescription);

        System.out.println("PRESCRIPTION CREATE: Auto scheduling TIME-BASED reminders for elder user: " + elderUser.getId());
        
        // AUTO SCHEDULE với TIME-BASED SCHEDULER thay vì cron cũ
        try {
            simpleTimeBasedScheduler.scheduleUserReminders(elderUser.getId());
            System.out.println("TIME-BASED scheduling completed for prescription: " + savedPrescription.getId());
        } catch (Exception e) {
            System.err.println("Prescription created but TIME-BASED scheduling failed: " + e.getMessage());
            e.printStackTrace();
        }

        List<MedicationReminderResponse> medicationReminderResponses =
                (savedPrescription.getMedicationReminders() == null) ? List.of() :
                        savedPrescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(savedPrescription, medicationReminderResponses);

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_CREATED)
                .data(response)
                .build();
    }

    @Transactional
    public BaseResponse<PrescriptionResponse> update(Long prescriptionId,
                                                     PrescriptionCreateRequest prescriptionUpdateRequest,
                                                     Authentication authentication) {
        System.out.println("Updating prescription ID: " + prescriptionId + ", Auth email: " + authentication.getName());
        
        // 1. Find existing prescription first
        Prescriptions existingPrescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        // 2. Validate authenticated user (could be Elder or Supervisor)
        // Try Elder first
        ElderUser authUserAsElder = elderUserRepository.findByEmail(authentication.getName()).orElse(null);
        SupervisorUser authUserAsSupervisor = supervisorUserRepository.findByEmail(authentication.getName()).orElse(null);

        // 3. Permission check: owner (elder) OR supervisor with update permission
        boolean isOwner = false;
        boolean isSupervisorWithPermission = false;

        if (authUserAsElder != null) {
            System.out.println("Auth user is Elder ID: " + authUserAsElder.getId());
            isOwner = existingPrescription.getElderUser().getId().equals(authUserAsElder.getId());
            System.out.println("Is owner: " + isOwner);
        }

        if (authUserAsSupervisor != null) {
            System.out.println("Auth user is Supervisor ID: " + authUserAsSupervisor.getId());
            isSupervisorWithPermission = elderSupervisorRepository
                    .findActiveWithUpdatePermission(authUserAsSupervisor.getId(), existingPrescription.getElderUser().getId())
                    .isPresent();
            System.out.println("Is supervisor with permission: " + isSupervisorWithPermission);
        }

        if (!isOwner && !isSupervisorWithPermission) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

//        // Validate new data
//        validateMedicationReminders(prescriptionUpdateRequest.getMedicationReminders());

        Long targetElderId = existingPrescription.getElderUser().getId();
        System.out.println("PRESCRIPTION UPDATE: Preparing to reschedule TIME-BASED reminders for elder user: " + targetElderId);

        // Lấy old reminders trước khi update
        List<MedicationReminder> oldReminders = existingPrescription.getMedicationReminders();

        //cập nhật thông tin prescription
        existingPrescription.setName(prescriptionUpdateRequest.getName());
        existingPrescription.setDescription(prescriptionUpdateRequest.getDescription());
        existingPrescription.setImageUrl(prescriptionUpdateRequest.getImageUrl());

        // tao mới medication reminders nếu có
        if (prescriptionUpdateRequest.getMedicationReminders() != null &&
                !prescriptionUpdateRequest.getMedicationReminders().isEmpty()) {

            // Delete old reminders
            if (oldReminders != null && !oldReminders.isEmpty()) {
                medicationReminderRepository.deleteAll(oldReminders);
            }

            // Create new reminders
            List<MedicationReminder> newReminders = prescriptionUpdateRequest.getMedicationReminders().stream()
                    .map(mr -> PrescriptionMapper.toEntities(mr, existingPrescription.getElderUser(), existingPrescription))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());

            existingPrescription.setMedicationReminders(newReminders);
        } else {
            // No new reminders - delete all old ones
            if (oldReminders != null && !oldReminders.isEmpty()) {
                medicationReminderRepository.deleteAll(oldReminders);
            }
            existingPrescription.setMedicationReminders(null);
        }

        Prescriptions updatedPrescription = prescriptionRepository.save(existingPrescription);

                System.out.println("PRESCRIPTION UPDATE: Auto rescheduling TIME-BASED reminders for elder user: " + targetElderId);
        
        // AUTO RESCHEDULE với TIME-BASED SCHEDULER thay vì tạo từng cron job
        try {
                        simpleTimeBasedScheduler.scheduleUserReminders(targetElderId);
                        System.out.println("TIME-BASED rescheduling completed for prescription: " + updatedPrescription.getId());
        } catch (Exception e) {
            System.err.println("Prescription updated but TIME-BASED rescheduling failed: " + e.getMessage());
            e.printStackTrace();
        }

        List<MedicationReminderResponse> medicationReminderResponses =
                (updatedPrescription.getMedicationReminders() == null) ? List.of() :
                        updatedPrescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(updatedPrescription, medicationReminderResponses);

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_UPDATED)
                .data(response)
                .build();
    }

    public BaseResponse<PrescriptionResponse> getById(Long prescriptionId, Authentication authentication) {
        System.out.println("=== getById START ===");
        System.out.println("Requested prescriptionId: " + prescriptionId);
        System.out.println("Authenticated user email: " + authentication.getName());
        
        // FIX: Use userRepository instead of elderUserRepository to support both Elder and Supervisor
        User authUser = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> {
                    System.err.println("Authenticated user not found: " + authentication.getName());
                    return new BaseException.EntityNotFoundException(USER_NOT_FOUND);
                });
        
        System.out.println("Auth user found - ID: " + authUser.getId() + ", Type: " + authUser.getClass().getSimpleName());

        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> {
                    System.err.println("Prescription not found with ID: " + prescriptionId);
                    return new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND);
                });
        
        System.out.println("Prescription found - ID: " + prescription.getId() + ", Owner: " + prescription.getElderUser().getId());

        // If not owner, allow supervisor with view permission
        boolean isOwner = prescription.getElderUser().getId().equals(authUser.getId());
        System.out.println("Is owner? " + isOwner);
        
        boolean isSupervisorWithView = false;
        if (authUser instanceof SupervisorUser) {
            System.out.println("Auth user is Supervisor - checking permission...");
            isSupervisorWithView = elderSupervisorRepository
                    .findActiveWithViewPermission(authUser.getId(), prescription.getElderUser().getId())
                    .isPresent();
            System.out.println("Has view permission? " + isSupervisorWithView);
        } else {
            System.out.println("Auth user is Elder (not Supervisor)");
        }

        if (!isOwner && !isSupervisorWithView) {
            System.err.println("PERMISSION DENIED - isOwner: " + isOwner + ", isSupervisorWithView: " + isSupervisorWithView);
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }
        
        System.out.println("Permission check passed");

        // Gộp medications (cùng name → array reminderTimes)
        List<MedicationResponse> medications = PrescriptionMapper.groupMedicationsByName(
                prescription.getMedicationReminders()
        );

        // Legacy field (deprecated)
        List<MedicationReminderResponse> medicationReminderResponses =
                (prescription.getMedicationReminders() == null) ? List.of() :
                        prescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionResponse.builder()
                .id(prescription.getId())
                .name(prescription.getName())
                .description(prescription.getDescription())
                .imageUrl(prescription.getImageUrl())
                .isActive(prescription.getIsActive())
                .userId(prescription.getElderUser().getId())
                .medications(medications)  // Grouped medications
                .medicationReminders(medicationReminderResponses)  // Legacy
                .build();

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_FETCHED)
                .data(response)
                .build();
    }

    public BaseResponse<List<PrescriptionResponse>> getAllByUser(Authentication authentication) {
        ElderUser elderUser = elderUserRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // Lấy TẤT CẢ đơn thuốc (cả active và inactive), sắp xếp theo ngày tạo mới nhất
        List<Prescriptions> prescriptions = prescriptionRepository.findByElderUser_IdOrderByCreatedAtDesc(elderUser.getId());

        List<PrescriptionResponse> prescriptionResponses = prescriptions.stream()
                .map(prescription -> {
                    // Gộp medications
                    List<MedicationResponse> medications = PrescriptionMapper.groupMedicationsByName(
                            prescription.getMedicationReminders()
                    );

                    // Legacy field
                    List<MedicationReminderResponse> medicationReminderResponses =
                            (prescription.getMedicationReminders() == null) ? List.of() :
                                    prescription.getMedicationReminders().stream()
                                            .map(PrescriptionMapper::toResponse)
                                            .toList();

                    return PrescriptionResponse.builder()
                            .id(prescription.getId())
                            .name(prescription.getName())
                            .description(prescription.getDescription())
                            .imageUrl(prescription.getImageUrl())
                            .isActive(prescription.getIsActive())
                            .userId(prescription.getElderUser().getId())
                            .medications(medications)  // Grouped
                            .medicationReminders(medicationReminderResponses)  // Legacy
                            .build();
                })
                .toList();

        return BaseResponse.<List<PrescriptionResponse>>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_FETCHED)
                .data(prescriptionResponses)
                .build();
    }

    /**
     * Lấy tất cả đơn thuốc của user theo userId (cho Supervisor xem đơn của Elder)
     * Endpoint này CẦN THIẾT vì:
     * - Supervisor muốn xem đơn thuốc của Elder (không phải của chính mình)
     * - getAllByUser() chỉ trả về đơn của user đang authenticated
     */
    public BaseResponse<List<PrescriptionResponse>> getAllByUserId(Long userId, Authentication authentication) {
        System.out.println("=== getAllByUserId START ===");
        System.out.println("Requested elderUserId: " + userId);
        System.out.println("Authenticated user email: " + authentication.getName());
        
        // Get authenticated user (could be Elder or Supervisor)
        User authUser = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> {
                    System.err.println("Authenticated user not found: " + authentication.getName());
                    return new BaseException.EntityNotFoundException(USER_NOT_FOUND);
                });
        
        System.out.println("Auth user found - ID: " + authUser.getId() + ", Type: " + authUser.getClass().getSimpleName());

        ElderUser elderUser = elderUserRepository.findById(userId)
                .orElseThrow(() -> {
                    System.err.println("Elder user not found with ID: " + userId);
                    return new BaseException.EntityNotFoundException("Elder user không tồn tại");
                });
        
        System.out.println("Elder user found - ID: " + elderUser.getId() + ", Name: " + elderUser.getFullName());

        // Check permission: must be owner or supervisor with view permission
        boolean isOwner = elderUser.getId().equals(authUser.getId());
        System.out.println("Is owner? " + isOwner);
        
        boolean isSupervisorWithView = false;
        if (authUser instanceof SupervisorUser) {
            System.out.println("Auth user is Supervisor - checking permission...");
            isSupervisorWithView = elderSupervisorRepository
                    .findActiveWithViewPermission(authUser.getId(), elderUser.getId())
                    .isPresent();
            System.out.println("Has view permission? " + isSupervisorWithView);
        } else {
            System.out.println("Auth user is Elder (not Supervisor)");
        }

        if (!isOwner && !isSupervisorWithView) {
            System.err.println("PERMISSION DENIED - isOwner: " + isOwner + ", isSupervisorWithView: " + isSupervisorWithView);
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }
        
        System.out.println("Permission check passed");


        // Lấy tất cả đơn thuốc của Elder
        List<Prescriptions> prescriptions = prescriptionRepository.findByElderUser_IdOrderByCreatedAtDesc(elderUser.getId());

        List<PrescriptionResponse> prescriptionResponses = prescriptions.stream()
                .map(prescription -> {
                    List<MedicationResponse> medications = PrescriptionMapper.groupMedicationsByName(
                            prescription.getMedicationReminders()
                    );

                    List<MedicationReminderResponse> medicationReminderResponses =
                            (prescription.getMedicationReminders() == null) ? List.of() :
                                    prescription.getMedicationReminders().stream()
                                            .map(PrescriptionMapper::toResponse)
                                            .toList();

                    return PrescriptionResponse.builder()
                            .id(prescription.getId())
                            .name(prescription.getName())
                            .description(prescription.getDescription())
                            .imageUrl(prescription.getImageUrl())
                            .isActive(prescription.getIsActive())
                            .userId(prescription.getElderUser().getId())
                            .medications(medications)
                            .medicationReminders(medicationReminderResponses)
                            .build();
                })
                .toList();

        return BaseResponse.<List<PrescriptionResponse>>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_FETCHED)
                .data(prescriptionResponses)
                .build();
    }

    @Transactional
    public BaseResponse<String> delete(Long prescriptionId, Authentication authentication) {
        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        // Try both Elder and Supervisor
        ElderUser authUserAsElder = elderUserRepository.findByEmail(authentication.getName()).orElse(null);
        SupervisorUser authUserAsSupervisor = supervisorUserRepository.findByEmail(authentication.getName()).orElse(null);

        // Allow owner or supervisor with update permission
        boolean isOwner = false;
        boolean isSupervisorWithPermission = false;

        if (authUserAsElder != null) {
            isOwner = prescription.getElderUser().getId().equals(authUserAsElder.getId());
            System.out.println("Is owner (Elder): " + isOwner);
        }

        if (authUserAsSupervisor != null) {
            isSupervisorWithPermission = elderSupervisorRepository
                    .findActiveWithUpdatePermission(authUserAsSupervisor.getId(), prescription.getElderUser().getId())
                    .isPresent();
            System.out.println("Is supervisor with permission: " + isSupervisorWithPermission);
        }

        if (!isOwner && !isSupervisorWithPermission) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

        System.out.println("PRESCRIPTION DELETE: Preparing to reschedule TIME-BASED reminders for elder user: " + prescription.getElderUser().getId());

        // Delete prescription (cascade sẽ xóa luôn medication reminders)
        prescriptionRepository.delete(prescription);

        // AUTO RESCHEDULE TIME-BASED reminders cho medications còn lại của user
                try {
                        simpleTimeBasedScheduler.scheduleUserReminders(prescription.getElderUser().getId());
                        System.out.println("TIME-BASED rescheduling completed after deleting prescription: " + prescriptionId);
                } catch (Exception e) {
                        System.err.println("Prescription deleted but TIME-BASED rescheduling failed: " + e.getMessage());
                        e.printStackTrace();
                }

        return BaseResponse.<String>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_DELETED)
                .data("Prescription " + prescriptionId + " deleted and reminders rescheduled")
                .build();
    }

    /**
     * Toggle trạng thái đơn thuốc (active <-> inactive)
     * Tự động cập nhật trạng thái tất cả medications và reschedule TIME-BASED jobs
     */
    @Transactional
    public BaseResponse<PrescriptionResponse> toggleStatus(Long prescriptionId, Authentication authentication) {
        // 1. Find prescription
        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        // 2. Validate authenticated user (Elder or Supervisor)
        ElderUser authUserAsElder = elderUserRepository.findByEmail(authentication.getName()).orElse(null);
        SupervisorUser authUserAsSupervisor = supervisorUserRepository.findByEmail(authentication.getName()).orElse(null);

        // 3. Permission: owner or supervisor with update permission
        boolean isOwner = false;
        boolean isSupervisorWithPermission = false;

        if (authUserAsElder != null) {
            isOwner = prescription.getElderUser().getId().equals(authUserAsElder.getId());
            System.out.println("Is owner (Elder): " + isOwner);
        }

        if (authUserAsSupervisor != null) {
            isSupervisorWithPermission = elderSupervisorRepository
                    .findActiveWithUpdatePermission(authUserAsSupervisor.getId(), prescription.getElderUser().getId())
                    .isPresent();
            System.out.println("Is supervisor with permission: " + isSupervisorWithPermission);
        }

        if (!isOwner && !isSupervisorWithPermission) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

        // 4. Toggle prescription status
        boolean newStatus = !prescription.getIsActive();
        prescription.setIsActive(newStatus);

        // 5. Toggle all medications status
        if (prescription.getMedicationReminders() != null) {
            prescription.getMedicationReminders().forEach(medication -> {
                medication.setIsActive(newStatus);
            });
        }

        // 6. Save changes
        Prescriptions updatedPrescription = prescriptionRepository.save(prescription);

                System.out.println("PRESCRIPTION TOGGLE STATUS: " + prescriptionId + " -> " + (newStatus ? "ACTIVE" : "INACTIVE"));

                // 7. AUTO RESCHEDULE TIME-BASED reminders
                try {
                        simpleTimeBasedScheduler.scheduleUserReminders(prescription.getElderUser().getId());
                        System.out.println("TIME-BASED rescheduling completed after toggling prescription: " + prescriptionId);
                } catch (Exception e) {
                        System.err.println("Prescription status toggled but TIME-BASED rescheduling failed: " + e.getMessage());
                        e.printStackTrace();
                }

        // 8. Build response - Bao gồm cả medications (grouped) và medicationReminders (legacy)
        List<MedicationResponse> medications = PrescriptionMapper.groupMedicationsByName(
                updatedPrescription.getMedicationReminders()
        );

        List<MedicationReminderResponse> medicationReminderResponses =
                (updatedPrescription.getMedicationReminders() == null) ? List.of() :
                        updatedPrescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionResponse.builder()
                .id(updatedPrescription.getId())
                .name(updatedPrescription.getName())
                .description(updatedPrescription.getDescription())
                .imageUrl(updatedPrescription.getImageUrl())
                .isActive(updatedPrescription.getIsActive())
                .userId(updatedPrescription.getElderUser().getId())
                .medications(medications)  // Grouped medications
                .medicationReminders(medicationReminderResponses)  // Legacy
                .build();

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message("Prescription status updated to " + (newStatus ? "active" : "inactive"))
                .data(response)
                .build();
    }

//    private void validateMedicationReminders(List<?> medicationReminders) {
//
//    }
}