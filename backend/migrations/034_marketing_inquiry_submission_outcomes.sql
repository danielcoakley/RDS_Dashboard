ALTER TABLE marketing_inquiry_submission_events
    ADD COLUMN outcome TEXT NOT NULL DEFAULT 'accepted' CHECK (
        outcome IN ('accepted', 'rate_limited')
    );

UPDATE marketing_inquiry_submission_events
SET outcome = 'rate_limited'
WHERE inquiry_id IS NULL;
