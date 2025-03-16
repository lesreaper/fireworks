import logging
from .types import EvaluationMetrics
from .BaseAgent import BaseAgent
from typing import Dict, Any
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SysEvalAgent(BaseAgent):
    def process(self, state: Dict[str, Any]) -> EvaluationMetrics:
        """Calculate detailed evaluation metrics by comparing with ground truth"""
        try:
            # Load ground truth data from file
            try:
                with open('./eval/actual_results.json', 'r') as f:
                    ground_truth_data = json.loads(f.read())
                logger.info(f"Successfully loaded ground truth data with {len(ground_truth_data)} entries")
            except Exception as e:
                logger.error(f"Failed to load ground truth data: {str(e)}")
                raise

            # Get the current file being processed
            image_path = state.get("image_path", "")
            file_name = os.path.basename(image_path)
            logger.info(f"Evaluating file: {file_name}")

            # Find matching ground truth
            ground_truth = None
            for entry in ground_truth_data:
                if entry["file"].replace(" ", "-") == file_name:  # Handle space/dash differences
                    ground_truth = entry["data"]
                    break

            if not ground_truth:
                logger.error(f"No ground truth found for file: {file_name}")
                return EvaluationMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    accuracy=0.0,
                    error_rate=1.0,
                    per_field_stats={},
                    processing_stats={},
                    token_usage={},
                    document_type="Unknown",
                    validation_status=False,
                    confidence_scores={},
                    error_messages=["No ground truth data found"]
                )

            extracted_data = state.get("extracted_data", {}) or {}

            # Calculate metrics
            field_stats = self._calculate_field_stats(ground_truth, extracted_data)
            metrics = self._calculate_overall_metrics(field_stats, state)

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def _calculate_field_stats(self, ground_truth: Dict, extracted_data: Dict) -> Dict:
        """Calculate per-field statistics"""
        field_stats = {}

        normalized_extracted = {
            "document_type": extracted_data.get("document_type"),
            "document_number": extracted_data.get("document_number"),
            "first_name": extracted_data.get("name", {}).get("first_name"),
            "last_name": extracted_data.get("name", {}).get("last_name"),
            "state": extracted_data.get("state"),
            "date_of_birth": extracted_data.get("date_of_birth"),
            "issue_date": extracted_data.get("issue_date"),
            "expiration_date": extracted_data.get("expiry_date"),
            "address": extracted_data.get("address", {}).get("street", "") + ", " + extracted_data.get("address", {}).get("city", "") + ", " + extracted_data.get("address", {}).get("state", "") + " " + extracted_data.get("address", {}).get("zip_code", "")
        }

        for field in ground_truth.keys():
            gt_value = str(ground_truth.get(field, "")).lower().strip()
            extracted_value = str(normalized_extracted.get(field, "")).lower().strip()

            has_gt = bool(gt_value)
            has_prediction = bool(extracted_value)
            is_correct = gt_value == extracted_value

            precision = 1.0 if (has_prediction and is_correct) else 0.0
            recall = 1.0 if (has_gt and is_correct) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            field_stats[field] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        return field_stats

    def _calculate_overall_metrics(self, field_stats: Dict, state: Dict) -> EvaluationMetrics:
        """Calculate overall metrics from field statistics"""
        total_fields = len(field_stats)
        total_correct = sum(1 for stats in field_stats.values() if stats["precision"] == 1.0 and stats["recall"] == 1.0)

        accuracy = total_correct / total_fields if total_fields > 0 else 0.0
        error_rate = 1.0 - accuracy

        # Calculate overall precision, recall, and F1
        precisions = [stats["precision"] for stats in field_stats.values()]
        recalls = [stats["recall"] for stats in field_stats.values()]

        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Get process statistics
        processing_stats = {
            "extraction_attempts": state.get("extraction_attempts", 0),
            "total_tokens": state.get("total_tokens", 0),
            "document_type": state.get("doc_type", "Unknown"),
            "validation_status": state.get("validation_status", False)
        }

        # Token usage
        token_usage = {
            "total": state.get("total_tokens", 0),
            "doc_type_detection": state.get("doc_type_tokens", 0),
            "extraction": state.get("extraction_tokens", 0),
            "validation": state.get("validation_tokens", 0)
        }

        # Confidence scores
        confidence_scores = {
            "document_type": state.get("doc_type_confidence", 0.0),
            "validation": state.get("validation_confidence", 0.0)
        }

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            error_rate=error_rate,
            per_field_stats=field_stats,
            processing_stats=processing_stats,
            token_usage=token_usage,
            document_type=state.get("doc_type", "Unknown"),
            validation_status=state.get("validation_status", False),
            confidence_scores=confidence_scores,
            error_messages=[state.get("error_message")] if state.get("error_message") else None
        )
