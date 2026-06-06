import json, csv, uuid, io, os, requests
from django.shortcuts    import render
from django.http         import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db.models    import Avg
from django.utils        import timezone
from datetime            import timedelta
from .models             import Transaction

  
_port = os.environ.get('PORT', '10000')
API = f'http://127.0.0.1:{_port}'
def _gen_txn_id():
    return 'TXN' + uuid.uuid4().hex[:8].upper()

def index(request):
    try:
        info = requests.get(f"{API}/api/model/info", timeout=5).json()
    except Exception:
        info = {}
    total    = Transaction.objects.count()
    frauds   = Transaction.objects.filter(is_fraud=True).count()
    legit    = Transaction.objects.filter(is_fraud=False).count()
    avg_risk = Transaction.objects.aggregate(a=Avg('fraud_probability'))['a'] or 0
    recent   = Transaction.objects.all()[:10]
    trend = []
    for i in range(6, -1, -1):
        day = timezone.now() - timedelta(days=i)
        trend.append({
            'date':  day.strftime('%b %d'),
            'total': Transaction.objects.filter(created_at__date=day.date()).count(),
            'fraud': Transaction.objects.filter(created_at__date=day.date(), is_fraud=True).count(),
        })
    risk_dist = {
        'LOW':      Transaction.objects.filter(risk_level='LOW').count(),
        'MEDIUM':   Transaction.objects.filter(risk_level='MEDIUM').count(),
        'HIGH':     Transaction.objects.filter(risk_level='HIGH').count(),
        'CRITICAL': Transaction.objects.filter(risk_level='CRITICAL').count(),
    }
    ctx = {
        'model_info': info, 'total': total, 'frauds': frauds, 'legit': legit,
        'avg_risk': round(avg_risk * 100, 1),
        'fraud_rate': round((frauds / total * 100) if total else 0, 2),
        'recent': recent,
        'trend_json': json.dumps(trend),
        'risk_json':  json.dumps(risk_dist),
    }
    return render(request, 'dashboard/index.html', ctx)

@csrf_exempt
@require_http_methods(['POST'])
def predict(request):
    try:
        payload = json.loads(request.body)
        resp    = requests.post(f"{API}/api/predict", json=payload, timeout=10)
        data    = resp.json()
        if 'error' not in data:
            Transaction.objects.create(
                transaction_id=_gen_txn_id(), amount=payload.get('amount', 0),
                hour=payload.get('hour', 0), is_fraud=data['is_fraud'],
                fraud_probability=data['fraud_probability'],
                risk_level=data['risk_level'], model_used=data.get('model_used', ''),
                explanation=data.get('explanation', ''),
            )
        return JsonResponse(data, status=resp.status_code)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(['POST'])
def batch_upload(request):
    try:
        csv_file = request.FILES.get('csv_file')
        if not csv_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        content = csv_file.read().decode('utf-8')
        rows    = list(csv.DictReader(io.StringIO(content)))
        if len(rows) > 1000:
            return JsonResponse({'error': 'Max 1000 rows allowed'}, status=400)
        results_list, fraud_ct = [], 0
        for row in rows:
            try:
                payload = {
                    'amount': float(row.get('Amount', row.get('amount', 0))),
                    'hour':   int(float(row.get('hour', 12))),
                }
                for i in range(1, 29):
                    payload[f'V{i}'] = float(row.get(f'V{i}', 0))
                resp = requests.post(f"{API}/api/predict", json=payload, timeout=10).json()
                if 'error' not in resp:
                    Transaction.objects.create(
                        transaction_id=_gen_txn_id(), amount=payload['amount'],
                        hour=payload['hour'], is_fraud=resp['is_fraud'],
                        fraud_probability=resp['fraud_probability'],
                        risk_level=resp['risk_level'],
                        model_used=resp.get('model_used', ''),
                        explanation=resp.get('explanation', ''),
                    )
                    if resp['is_fraud']: fraud_ct += 1
                    results_list.append({
                        'amount': payload['amount'], 'is_fraud': resp['is_fraud'],
                        'risk_level': resp['risk_level'], 'probability': resp['fraud_probability'],
                    })
            except Exception:
                continue
        return JsonResponse({
            'total': len(results_list), 'frauds_found': fraud_ct,
            'fraud_rate': round(fraud_ct/len(results_list)*100,2) if results_list else 0,
            'results': results_list[:50],
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def transaction_history(request):
    page, per_page = int(request.GET.get('page', 1)), 20
    qs = Transaction.objects.all()
    if request.GET.get('risk'): qs = qs.filter(risk_level=request.GET['risk'].upper())
    if request.GET.get('fraud') == '1': qs = qs.filter(is_fraud=True)
    total = qs.count()
    txns  = list(qs[(page-1)*per_page:page*per_page].values(
        'transaction_id','amount','hour','is_fraud',
        'fraud_probability','risk_level','model_used','created_at'))
    for t in txns:
        t['created_at']        = t['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        t['fraud_probability'] = round(t['fraud_probability']*100, 1)
    return JsonResponse({'transactions': txns, 'total': total, 'page': page,
                         'pages': (total+per_page-1)//per_page})

def model_info(request):
    try:
        info = requests.get(f"{API}/api/model/info", timeout=5).json()
        return JsonResponse({'info': info})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=503)

def export_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="transactions.csv"'
    writer = csv.writer(response)
    writer.writerow(['ID','Amount','Hour','Fraud','Probability%','Risk','Model','Time'])
    for t in Transaction.objects.all():
        writer.writerow([t.transaction_id, t.amount, t.hour, t.is_fraud,
                         round(t.fraud_probability*100,1), t.risk_level,
                         t.model_used, t.created_at.strftime('%Y-%m-%d %H:%M')])
    return response
