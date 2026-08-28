ALTER TABLE organization_evidence_package_templates
ADD COLUMN section_labels_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE organization_evidence_package_templates
ADD COLUMN section_order_json TEXT NOT NULL DEFAULT '[]';
