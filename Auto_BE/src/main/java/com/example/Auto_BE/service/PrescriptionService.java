//package com.example.Auto_BE.service;
//
//import com.example.Auto_BE.dto.BaseResponse;
//import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
//import com.example.Auto_BE.dto.response.MedicationReminderResponse;
//import com.example.Auto_BE.dto.response.PrescriptionResponse;
//import com.example.Auto_BE.entity.MedicationReminder;
//import com.example.Auto_BE.entity.Prescriptions;
//import com.example.Auto_BE.entity.User;
//import com.example.Auto_BE.exception.BaseException;
//import com.example.Auto_BE.mapper.PrescriptionMapper;
//import com.example.Auto_BE.repository.MedicationReminderRepository;
//import com.example.Auto_BE.repository.PrescriptionRepository;
//import com.example.Auto_BE.repository.UserRepository;
//import jakarta.transaction.Transactional;
//import org.springframework.security.core.Authentication;
//import org.springframework.stereotype.Service;
//
//import java.util.List;
//import java.util.stream.Collectors;
//
//import static com.example.Auto_BE.constants.ErrorMessages.USER_NOT_FOUND;
//import static com.example.Auto_BE.constants.SuccessMessage.PRESCRIPTION_CREATED;
//import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;
//
//@Service
//public class PrescriptionService {
//    private final UserRepository userRepository;
//    private final PrescriptionRepository prescriptionRepository;
//    private final MedicationReminderRepository medicationReminderRepository;
//    private final SchedulerReminder schedulerReminder;
//
//    public PrescriptionService(UserRepository userRepository,
//                               PrescriptionRepository prescriptionRepository,
//                               MedicationReminderRepository medicationReminderRepository, SchedulerReminder schedulerReminder) {
//        this.userRepository = userRepository;
//        this.prescriptionRepository = prescriptionRepository;
//        this.medicationReminderRepository = medicationReminderRepository;
//        this.schedulerReminder = schedulerReminder;
//    }
//
//    @Transactional
//    public BaseResponse<PrescriptionResponse> create(PrescriptionCreateRequest prescriptionCreateRequest,
//                                                     Authentication authentication) {
//        // Kiểm tra người dùng có tồn tại không
//        User user = userRepository.findByEmail(authentication.getName())
//                .orElseThrow(() -> new BaseException.EntityNotFoundException(USER_NOT_FOUND));
//
//        Prescriptions prescription = PrescriptionMapper.toEntity(prescriptionCreateRequest, user);
//
//        if(prescriptionCreateRequest.getMedicationReminders()!=null && !prescriptionCreateRequest.getMedicationReminders().isEmpty()) {
//            List<MedicationReminder> mrs = prescriptionCreateRequest.getMedicationReminders().stream()
//                    .map(mr -> PrescriptionMapper.toEntities(mr, user, prescription))
//                    .flatMap(List::stream)
//                    .collect(Collectors.toList());
//            prescription.setMedicationReminders(mrs);
//        }
//
//        Prescriptions savedPrescription = prescriptionRepository.save(prescription);
//
//         // ===== THÊM PHẦN TẠO LỊCH NGAY LẬP TỨC =====
//        if (savedPrescription.getMedicationReminders() != null && !savedPrescription.getMedicationReminders().isEmpty()) {
//            System.out.println("=== Scheduling reminders for new prescription ===");
//            System.out.println("Prescription ID: " + savedPrescription.getId());
//            System.out.println("Number of medication reminders: " + savedPrescription.getMedicationReminders().size());
//
//            for (MedicationReminder reminder : savedPrescription.getMedicationReminders()) {
//                try {
//                    System.out.println("Creating schedule for reminder: " + reminder.getName() + " (ID: " + reminder.getId() + ")");
//                    schedulerReminder.scheduleImmediatelyForNewReminder(reminder.getId());
//                    System.out.println("Successfully scheduled reminder ID: " + reminder.getId());
//                } catch (Exception e) {
//                    System.err.println("ERROR scheduling reminder ID " + reminder.getId() + ": " + e.getMessage());
//                    // Log error nhưng không throw exception để không ảnh hưởng đến việc tạo prescription
//                    e.printStackTrace();
//                }
//            }
//            System.out.println("=== Finished scheduling reminders ===");
//        } else {
//            System.out.println("No medication reminders to schedule for prescription ID: " + savedPrescription.getId());
//        }
//
//        List<MedicationReminderResponse> medicationReminderResponses =(savedPrescription.getMedicationReminders()==null)? List.of():
//                savedPrescription.getMedicationReminders().stream()
//                        .map(PrescriptionMapper::toResponse)
//                        .toList();
//
//        PrescriptionResponse response = PrescriptionMapper.toResponse(savedPrescription, medicationReminderResponses);
//        return BaseResponse.<PrescriptionResponse>builder()
//                .status(SUCCESS)
//                .message(PRESCRIPTION_CREATED)
//                .data(response)
//                .build();
//    }
//}
