package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.MedicationReminderResponse;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Prescriptions;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.mapper.PrescriptionMapper;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import com.example.Auto_BE.repository.PrescriptionRepository;
import com.example.Auto_BE.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

import static com.example.Auto_BE.constants.ErrorMessages.USER_NOT_FOUND;
import static com.example.Auto_BE.constants.SuccessMessage.PRESCRIPTION_CREATED;
import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@Service
public class CronPrescriptionService {
    private final UserRepository userRepository;
    private final PrescriptionRepository prescriptionRepository;
    private final MedicationReminderRepository medicationReminderRepository;
    private final CronSchedulerService cronSchedulerService;

    public CronPrescriptionService(UserRepository userRepository,
                                   PrescriptionRepository prescriptionRepository,
                                   MedicationReminderRepository medicationReminderRepository,
                                   CronSchedulerService cronSchedulerService) {
        this.userRepository = userRepository;
        this.prescriptionRepository = prescriptionRepository;
        this.medicationReminderRepository = medicationReminderRepository;
        this.cronSchedulerService = cronSchedulerService;
    }

    @Transactional
    public BaseResponse<PrescriptionResponse> create(PrescriptionCreateRequest prescriptionCreateRequest,
                                                     Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

//        // 2. Validate medication reminders data
//        validateMedicationReminders(prescriptionCreateRequest.getMedicationReminders());

        Prescriptions prescription = PrescriptionMapper.toEntity(prescriptionCreateRequest, user);

        if (prescriptionCreateRequest.getMedicationReminders() != null && 
            !prescriptionCreateRequest.getMedicationReminders().isEmpty()) {
            
            List<MedicationReminder> medicationReminders = prescriptionCreateRequest.getMedicationReminders().stream()
                    .map(mr -> PrescriptionMapper.toEntities(mr, user, prescription))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());
            
            prescription.setMedicationReminders(medicationReminders);
        }

        Prescriptions savedPrescription = prescriptionRepository.save(prescription);

        // 6. ===== TẠO CRON JOBS THAY VÌ INDIVIDUAL JOBS =====
        if (savedPrescription.getMedicationReminders() != null && 
            !savedPrescription.getMedicationReminders().isEmpty()) {
            
            System.out.println("=== Creating cron schedules for new prescription ===");
            System.out.println("Prescription ID: " + savedPrescription.getId());
            System.out.println("Number of medication reminders: " + savedPrescription.getMedicationReminders().size());
            
            for (MedicationReminder reminder : savedPrescription.getMedicationReminders()) {
                try {
                    System.out.println("Creating cron schedule for: " + reminder.getName() + " (ID: " + reminder.getId() + ")");
                    cronSchedulerService.scheduleWithCron(reminder.getId());
                    System.out.println("Successfully created cron schedule for reminder ID: " + reminder.getId());
                } catch (Exception e) {
                    System.err.println("ERROR creating cron schedule for reminder ID " + reminder.getId() + ": " + e.getMessage());
                    e.printStackTrace();
                }
            }
            System.out.println("=== Finished creating cron schedules ===");
        } else {
            System.out.println("No medication reminders to schedule for prescription ID: " + savedPrescription.getId());
        }

        // 7. Build response
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
        // 1. Validate user
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // 2. Find existing prescription
        Prescriptions existingPrescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException("Prescription not found"));

        // 3. Check ownership
        if (!existingPrescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException("You don't have permission to update this prescription");
        }

//        // 4. Validate new data
//        validateMedicationReminders(prescriptionUpdateRequest.getMedicationReminders());

        System.out.println("=== Updating prescription ===");
        System.out.println("Prescription ID: " + prescriptionId);

        // 5. ===== HỦY TẤT CẢ CRON JOBS CŨ =====
        List<MedicationReminder> oldReminders = existingPrescription.getMedicationReminders();
        if (oldReminders != null && !oldReminders.isEmpty()) {
            System.out.println("Cancelling cron schedules for " + oldReminders.size() + " old reminders");
            for (MedicationReminder oldReminder : oldReminders) {
                try {
                    cronSchedulerService.cancelCronSchedule(oldReminder.getId());
                } catch (Exception e) {
                    System.err.println("Error cancelling cron schedule for reminder ID " + oldReminder.getId() + ": " + e.getMessage());
                }
            }
        }

        // 6. Update prescription entity
        existingPrescription.setName(prescriptionUpdateRequest.getName());
        existingPrescription.setDescription(prescriptionUpdateRequest.getDescription());
        existingPrescription.setImageUrl(prescriptionUpdateRequest.getImageUrl());

        // 7. ===== TẠO MEDICATION REMINDERS MỚI =====
        if (prescriptionUpdateRequest.getMedicationReminders() != null && 
            !prescriptionUpdateRequest.getMedicationReminders().isEmpty()) {
            
            // Delete old reminders
            if (oldReminders != null && !oldReminders.isEmpty()) {
                medicationReminderRepository.deleteAll(oldReminders);
            }

            // Create new reminders
            List<MedicationReminder> newReminders = prescriptionUpdateRequest.getMedicationReminders().stream()
                    .map(mr -> PrescriptionMapper.toEntities(mr, user, existingPrescription))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());
            
            existingPrescription.setMedicationReminders(newReminders);
        } else {
            // No new reminders
            if (oldReminders != null && !oldReminders.isEmpty()) {
                medicationReminderRepository.deleteAll(oldReminders);
            }
            existingPrescription.setMedicationReminders(null);
        }

        // 8. Save updated prescription
        Prescriptions updatedPrescription = prescriptionRepository.save(existingPrescription);

        // 9. ===== TẠO CRON JOBS MỚI =====
        if (updatedPrescription.getMedicationReminders() != null && 
            !updatedPrescription.getMedicationReminders().isEmpty()) {
            
            System.out.println("Creating cron schedules for new reminders");
            for (MedicationReminder newReminder : updatedPrescription.getMedicationReminders()) {
                try {
                    cronSchedulerService.scheduleWithCron(newReminder.getId());
                    System.out.println("Created cron schedule for new reminder ID: " + newReminder.getId());
                } catch (Exception e) {
                    System.err.println("Error creating cron schedule for new reminder ID " + newReminder.getId() + ": " + e.getMessage());
                }
            }
        }

        System.out.println("=== Finished updating prescription ===");

        // 10. Build response
        List<MedicationReminderResponse> medicationReminderResponses = 
                (updatedPrescription.getMedicationReminders() == null) ? List.of() :
                updatedPrescription.getMedicationReminders().stream()
                        .map(PrescriptionMapper::toResponse)
                        .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(updatedPrescription, medicationReminderResponses);
        
        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message("Prescription updated successfully")
                .data(response)
                .build();
    }

    public BaseResponse<PrescriptionResponse> getById(Long prescriptionId, Authentication authentication) {
        // 1. Validate user
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // 2. Find prescription
        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException("Prescription not found"));

        // 3. Check ownership
        if (!prescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException("You don't have permission to view this prescription");
        }

        // 4. Build response
        List<MedicationReminderResponse> medicationReminderResponses = 
                (prescription.getMedicationReminders() == null) ? List.of() :
                prescription.getMedicationReminders().stream()
                        .map(PrescriptionMapper::toResponse)
                        .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(prescription, medicationReminderResponses);
        
        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message("Prescription retrieved successfully")
                .data(response)
                .build();
    }

    public BaseResponse<List<PrescriptionResponse>> getAllByUser(Authentication authentication) {
        // 1. Validate user
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // 2. Find all prescriptions for user
        List<Prescriptions> prescriptions = prescriptionRepository.findByUserIdAndIsActiveTrue(user.getId());

        // 3. Build response
        List<PrescriptionResponse> prescriptionResponses = prescriptions.stream()
                .map(prescription -> {
                    List<MedicationReminderResponse> medicationReminderResponses = 
                            (prescription.getMedicationReminders() == null) ? List.of() :
                            prescription.getMedicationReminders().stream()
                                    .map(PrescriptionMapper::toResponse)
                                    .toList();
                    
                    return PrescriptionMapper.toResponse(prescription, medicationReminderResponses);
                })
                .toList();

        return BaseResponse.<List<PrescriptionResponse>>builder()
                .status(SUCCESS)
                .message("Prescriptions retrieved successfully")
                .data(prescriptionResponses)
                .build();
    }

    @Transactional
    public BaseResponse<String> delete(Long prescriptionId, Authentication authentication) {
        // 1. Validate user
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // 2. Find prescription
        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException("Prescription not found"));

        // 3. Check ownership
        if (!prescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException("You don't have permission to delete this prescription");
        }

        // 4. ===== HỦY TẤT CẢ CRON JOBS =====
        if (prescription.getMedicationReminders() != null && !prescription.getMedicationReminders().isEmpty()) {
            System.out.println("Cancelling cron schedules for deleted prescription");
            for (MedicationReminder reminder : prescription.getMedicationReminders()) {
                try {
                    cronSchedulerService.cancelCronSchedule(reminder.getId());
                } catch (Exception e) {
                    System.err.println("Error cancelling cron schedule: " + e.getMessage());
                }
            }
        }

        // 5. Delete prescription (cascade will delete medication reminders)
        prescriptionRepository.delete(prescription);

        return BaseResponse.<String>builder()
                .status(SUCCESS)
                .message("Prescription deleted successfully")
                .data("Prescription ID: " + prescriptionId)
                .build();
    }

//    private void validateMedicationReminders(List<?> medicationReminders) {
//
//    }
}
