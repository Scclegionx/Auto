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

import static com.example.Auto_BE.constants.ErrorMessages.*;
import static com.example.Auto_BE.constants.SuccessMessage.*;

@Service
public class CronPrescriptionService {
    private final UserRepository userRepository;
    private final PrescriptionRepository prescriptionRepository;
    private final MedicationReminderRepository medicationReminderRepository;
    private final SimpleTimeBasedScheduler simpleTimeBasedScheduler; // NEW - TIME-BASED SCHEDULING

    public CronPrescriptionService(UserRepository userRepository,
                                   PrescriptionRepository prescriptionRepository,
                                   MedicationReminderRepository medicationReminderRepository,
                                   SimpleTimeBasedScheduler simpleTimeBasedScheduler) {
        this.userRepository = userRepository;
        this.prescriptionRepository = prescriptionRepository;
        this.medicationReminderRepository = medicationReminderRepository;
        this.simpleTimeBasedScheduler = simpleTimeBasedScheduler;
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

        System.out.println("🎯 PRESCRIPTION CREATE: Auto scheduling TIME-BASED reminders for user: " + user.getId());
        
        // AUTO SCHEDULE với TIME-BASED SCHEDULER thay vì cron cũ
        try {
            simpleTimeBasedScheduler.scheduleUserReminders(user.getId());
            System.out.println("✅ TIME-BASED scheduling completed for prescription: " + savedPrescription.getId());
        } catch (Exception e) {
            System.err.println("⚠️ Prescription created but TIME-BASED scheduling failed: " + e.getMessage());
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
        // 1. Validate user
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        // 2. Find existing prescription
        Prescriptions existingPrescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        if (!existingPrescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

//        // Validate new data
//        validateMedicationReminders(prescriptionUpdateRequest.getMedicationReminders());

        System.out.println("🎯 PRESCRIPTION UPDATE: Preparing to reschedule TIME-BASED reminders for user: " + user.getId());
        
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
                    .map(mr -> PrescriptionMapper.toEntities(mr, user, existingPrescription))
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

        System.out.println("🎯 PRESCRIPTION UPDATE: Auto rescheduling TIME-BASED reminders for user: " + user.getId());
        
        // AUTO RESCHEDULE với TIME-BASED SCHEDULER thay vì tạo từng cron job
        try {
            simpleTimeBasedScheduler.scheduleUserReminders(user.getId());
            System.out.println("✅ TIME-BASED rescheduling completed for prescription: " + updatedPrescription.getId());
        } catch (Exception e) {
            System.err.println("⚠️ Prescription updated but TIME-BASED rescheduling failed: " + e.getMessage());
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
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        if (!prescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

        List<MedicationReminderResponse> medicationReminderResponses =
                (prescription.getMedicationReminders() == null) ? List.of() :
                        prescription.getMedicationReminders().stream()
                                .map(PrescriptionMapper::toResponse)
                                .toList();

        PrescriptionResponse response = PrescriptionMapper.toResponse(prescription, medicationReminderResponses);

        return BaseResponse.<PrescriptionResponse>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_FETCHED)
                .data(response)
                .build();
    }

    public BaseResponse<List<PrescriptionResponse>> getAllByUser(Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        List<Prescriptions> prescriptions = prescriptionRepository.findByUserIdAndIsActiveTrue(user.getId());

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
                .message(PRESCRIPTION_FETCHED)
                .data(prescriptionResponses)
                .build();
    }

    @Transactional
    public BaseResponse<String> delete(Long prescriptionId, Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));

        Prescriptions prescription = prescriptionRepository.findById(prescriptionId)
                .orElseThrow(() -> new BaseException.EntityNotFoundException(PRESCRIPTION_NOT_FOUND));

        if (!prescription.getUser().getId().equals(user.getId())) {
            throw new BaseException.BadRequestException(PERMISSION_ERROR);
        }

        System.out.println("🎯 PRESCRIPTION DELETE: Preparing to reschedule TIME-BASED reminders for user: " + user.getId());

        // Delete prescription (cascade sẽ xóa luôn medication reminders)
        prescriptionRepository.delete(prescription);

        // AUTO RESCHEDULE TIME-BASED reminders cho medications còn lại của user
        try {
            simpleTimeBasedScheduler.scheduleUserReminders(user.getId());
            System.out.println("✅ TIME-BASED rescheduling completed after deleting prescription: " + prescriptionId);
        } catch (Exception e) {
            System.err.println("⚠️ Prescription deleted but TIME-BASED rescheduling failed: " + e.getMessage());
            e.printStackTrace();
        }

        return BaseResponse.<String>builder()
                .status(SUCCESS)
                .message(PRESCRIPTION_DELETED)
                .data("Prescription " + prescriptionId + " deleted and reminders rescheduled")
                .build();
    }

//    private void validateMedicationReminders(List<?> medicationReminders) {
//
//    }
}