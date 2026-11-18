"""DICOM de-identification (subset of the DICOM PS3.15 Basic Profile).

Removes or blanks patient/staff/institution identifiers, replaces UIDs,
strips private (vendor) tags, and flags burned-in text, which tag cleaning
cannot remove because it is part of the pixels.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pydicom
from pydicom.uid import generate_uid

# Tags removed entirely.
REMOVE_TAGS = (
    "PatientName", "PatientID", "OtherPatientIDs", "OtherPatientNames",
    "OtherPatientIDsSequence", "PatientBirthDate", "PatientBirthTime",
    "PatientAddress", "PatientTelephoneNumbers", "PatientMotherBirthName",
    "MilitaryRank", "EthnicGroup", "Occupation", "AdditionalPatientHistory",
    "PatientComments", "MedicalRecordLocator", "ReferringPhysicianName",
    "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "OperatorsName",
    "PhysiciansOfRecord", "RequestingPhysician", "InstitutionName",
    "InstitutionAddress", "InstitutionalDepartmentName", "StationName",
    "DeviceSerialNumber", "AccessionNumber", "StudyID", "RequestAttributesSequence",
    "IssuerOfPatientID", "InsurancePlanIdentification",
)
# Dates are blanked to year-only (Safe Harbor keeps the year).
DATE_TAGS = ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")
TIME_TAGS = ("StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime")
UID_TAGS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID")


@dataclass(frozen=True)
class DicomDeidReport:
    removed_tags: list[str] = field(default_factory=list)
    private_tags_removed: bool = False
    burned_in_text_risk: bool = False
    warnings: list[str] = field(default_factory=list)


def anonymize_dicom(content: bytes) -> tuple[bytes, DicomDeidReport]:
    ds = pydicom.dcmread(io.BytesIO(content))
    removed: list[str] = []

    for tag in REMOVE_TAGS:
        if tag in ds:
            delattr(ds, tag)
            removed.append(tag)
    for tag in DATE_TAGS:
        if tag in ds and ds.get(tag):
            ds.data_element(tag).value = f"{str(ds.get(tag))[:4]}0101"
            removed.append(f"{tag}(month/day)")
    for tag in TIME_TAGS:
        if tag in ds:
            ds.data_element(tag).value = ""
            removed.append(tag)

    uid_map: dict[str, str] = {}
    for tag in UID_TAGS:
        if tag in ds:
            old = str(ds.get(tag))
            ds.data_element(tag).value = uid_map.setdefault(old, generate_uid())
            removed.append(f"{tag}(replaced)")
    if hasattr(ds, "file_meta") and "MediaStorageSOPInstanceUID" in ds.file_meta and "SOPInstanceUID" in ds:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    ds.remove_private_tags()
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "MedAssist basic profile (PS3.15 subset)"

    warnings = []
    burned_in = str(ds.get("BurnedInAnnotation", "")).upper() == "YES" or ds.get("Modality") in {"US", "SC", "OT"}
    if burned_in:
        warnings.append(
            "Image may contain burned-in text (identifiers in the pixels). "
            "Tag removal does not cover this; review the image before sharing."
        )

    out = io.BytesIO()
    ds.save_as(out, enforce_file_format=False)
    return out.getvalue(), DicomDeidReport(
        removed_tags=removed,
        private_tags_removed=True,
        burned_in_text_risk=burned_in,
        warnings=warnings,
    )
