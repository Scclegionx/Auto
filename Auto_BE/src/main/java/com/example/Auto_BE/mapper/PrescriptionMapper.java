package com.example.Auto_BE.mapper;


import com.example.Auto_BE.dto.request.MedicationReminderCreateRequest;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.MedicationReminderResponse;
import com.example.Auto_BE.dto.response.MedicationResponse;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Prescriptions;

import java.util.*;
import java.util.stream.Collectors;

public class PrescriptionMapper {

    public static Prescriptions toEntity(PrescriptionCreateRequest dto, ElderUser elderUser) {
        Prescriptions p = new Prescriptions()
                .setName(dto.getName())
                .setDescription(dto.getDescription())
                .setImageUrl(dto.getImageUrl())
                .setIsActive(true)
                .setElderUser(elderUser);
        return p;
    }

    // Mapper snippet (đặt trong PrescriptionMapper hoặc class tiện ích của bạn)
    public static List<MedicationReminder> toEntities(
            MedicationReminderCreateRequest dto,
            ElderUser elderUser,
            Prescriptions p
    ) {
        if (dto.getReminderTimes() == null || dto.getReminderTimes().isEmpty()) {
            return List.of();
        }

        // loại null/blank, unique & sort
        List<String> times = dto.getReminderTimes().stream()
                .filter(t -> t != null && !t.isBlank())
                .map(String::trim)
                .distinct()
                .sorted()
                .toList();
        List<MedicationReminder> list = new ArrayList<>(times.size());
        for (String timeStr : times) {
            MedicationReminder mr = new MedicationReminder()
                    .setName(dto.getName())
                    .setDescription(dto.getDescription())
                    .setType(dto.getType())
                    .setReminderTime(timeStr)              // mỗi bản ghi 1 giờ
                    .setDaysOfWeek(dto.getDaysOfWeek())
                    .setIsActive(true)
                    .setElderUser(elderUser)
                    .setPrescription(p);
            list.add(mr);
        }
        return list;
    }


    public static PrescriptionResponse toResponse(Prescriptions p, List<MedicationReminderResponse> reminders) {
        return PrescriptionResponse.builder()
                .id(p.getId())
                .name(p.getName())
                .description(p.getDescription())
                .imageUrl(p.getImageUrl())
                .isActive(p.getIsActive())
                .userId(p.getElderUser() != null ? p.getElderUser().getId() : null)
                .medicationReminders(reminders)
                .build();
    }

    public static MedicationReminderResponse toResponse(MedicationReminder mr) {
        return MedicationReminderResponse.builder()
                .id(mr.getId())
                .name(mr.getName())
                .description(mr.getDescription())
                .type(mr.getType())
                .reminderTime(mr.getReminderTime())
                .daysOfWeek(mr.getDaysOfWeek())
                .isActive(mr.getIsActive())
                .prescriptionId(mr.getPrescription() != null ? mr.getPrescription().getId() : null)
                .userId(mr.getElderUser() != null ? mr.getElderUser().getId() : null)
                .build();
    }

    /**
     * Gộp các MedicationReminder thành MedicationResponse
     * - Medications có cùng name+description+type+daysOfWeek → gộp reminderTimes thành array
     */
    public static List<MedicationResponse> groupMedicationsByName(List<MedicationReminder> medications) {
        if (medications == null || medications.isEmpty()) {
            return List.of();
        }

        // Group by: name + description + type + daysOfWeek
        Map<String, List<MedicationReminder>> grouped = medications.stream()
                .collect(Collectors.groupingBy(med -> 
                    med.getName() + "|" + 
                    (med.getDescription() != null ? med.getDescription() : "") + "|" + 
                    med.getType() + "|" + 
                    med.getDaysOfWeek()
                ));

        return grouped.values().stream()
                .map(group -> {
                    MedicationReminder first = group.get(0);
                    
                    // Lấy tất cả reminderTimes và sắp xếp
                    List<String> times = group.stream()
                            .map(MedicationReminder::getReminderTime)
                            .filter(Objects::nonNull)
                            .sorted()
                            .distinct()
                            .collect(Collectors.toList());

                    return MedicationResponse.builder()
                            .id(first.getId()) // Lấy ID của bản ghi đầu tiên (có thể dùng cho edit)
                            .medicationName(first.getName())
                            .description(first.getDescription())
                            .type(first.getType())
                            .reminderTimes(times)
                            .daysOfWeek(first.getDaysOfWeek())
                            .isActive(first.getIsActive())
                            .prescriptionId(first.getPrescription() != null ? first.getPrescription().getId() : null)
                            .userId(first.getElderUser() != null ? first.getElderUser().getId() : null)
                            .build();
                })
                .collect(Collectors.toList());
    }
}