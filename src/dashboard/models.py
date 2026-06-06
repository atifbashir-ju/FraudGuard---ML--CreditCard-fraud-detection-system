from django.db import models


class Transaction(models.Model):
    RISK_CHOICES = [
        ('LOW',      'Low'),
        ('MEDIUM',   'Medium'),
        ('HIGH',     'High'),
        ('CRITICAL', 'Critical'),
    ]

    transaction_id    = models.CharField(max_length=20, unique=True)
    amount            = models.FloatField()
    hour              = models.IntegerField()
    is_fraud          = models.BooleanField()
    fraud_probability = models.FloatField()
    risk_level        = models.CharField(max_length=10, choices=RISK_CHOICES)
    model_used        = models.CharField(max_length=50)
    explanation       = models.TextField(blank=True)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.transaction_id} | ${self.amount} | {self.risk_level}"
