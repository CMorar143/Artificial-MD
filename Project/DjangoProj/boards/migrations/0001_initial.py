# Generated by Django 2.2.7 on 2020-03-15 21:27

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Ailment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('description', models.CharField(max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Allergy',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Examination',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('height', models.FloatField()),
                ('weight', models.FloatField()),
                ('reg_pulse', models.BooleanField()),
                ('cp', models.PositiveIntegerField()),
                ('protein', models.FloatField()),
                ('hdl_chol', models.FloatField(null=True)),
                ('ldl_chol', models.FloatField(null=True)),
                ('triglyceride', models.FloatField()),
                ('uric_acid', models.FloatField()),
                ('heart_rate', models.PositiveIntegerField()),
                ('smoke_per_day', models.PositiveIntegerField()),
                ('smoker_years', models.PositiveIntegerField()),
                ('blood_systolic', models.FloatField()),
                ('blood_diastolic', models.FloatField()),
                ('chol_overall', models.FloatField()),
                ('fasting_glucose', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='Family',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('family_name', models.CharField(max_length=30)),
                ('family_hist', models.CharField(max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Medication',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('description', models.CharField(max_length=150)),
            ],
        ),
        migrations.CreateModel(
            name='Patient',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patient_name', models.CharField(max_length=50)),
                ('DOB', models.DateField()),
                ('age', models.PositiveIntegerField()),
                ('sex', models.PositiveIntegerField()),
                ('address_line1', models.CharField(max_length=40, null=True)),
                ('address_line2', models.CharField(max_length=40, null=True)),
                ('address_line3', models.CharField(max_length=40, null=True)),
                ('occupation', models.CharField(max_length=30)),
                ('marital_status', models.PositiveIntegerField()),
                ('acc_balance', models.PositiveIntegerField(default=0)),
                ('tel_num', models.PositiveIntegerField(null=True)),
                ('home_num', models.PositiveIntegerField(null=True)),
                ('next_app', models.DateTimeField(null=True)),
                ('recall_period', models.PositiveIntegerField(null=True)),
                ('family', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='boards.Family')),
            ],
        ),
        migrations.CreateModel(
            name='Visit',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('reason', models.CharField(max_length=300, null=True)),
                ('doctor_notes', models.CharField(max_length=100, null=True)),
                ('outcome', models.CharField(max_length=50, null=True)),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
        migrations.CreateModel(
            name='Reminder',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rem_date', models.DateField()),
                ('location', models.CharField(max_length=50, null=True)),
                ('message', models.CharField(max_length=150)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
        migrations.CreateModel(
            name='Patient_Medication',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('medication', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Medication')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
        migrations.CreateModel(
            name='Patient_Allergy',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('allergy', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Allergy')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
        migrations.CreateModel(
            name='Patient_Ailment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ailment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Ailment')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
        migrations.CreateModel(
            name='Medical_history',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('heart_attack', models.BooleanField()),
                ('angina', models.BooleanField()),
                ('breathlessness', models.BooleanField()),
                ('chest_pain', models.PositiveIntegerField()),
                ('high_chol', models.BooleanField()),
                ('high_bp', models.BooleanField()),
                ('hoarseness', models.BooleanField()),
                ('wheezing', models.BooleanField()),
                ('sweating', models.BooleanField()),
                ('diabetes', models.BooleanField()),
                ('stressed', models.BooleanField()),
                ('childhood_illness', models.CharField(max_length=20, null=True)),
                ('exam', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='boards.Examination')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
        migrations.CreateModel(
            name='Investigation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('further_actions', models.CharField(max_length=150)),
                ('ref_to', models.CharField(max_length=100, null=True)),
                ('ref_reason', models.CharField(max_length=150, null=True)),
                ('result', models.CharField(max_length=200, null=True)),
                ('visit', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Visit')),
            ],
        ),
        migrations.AddField(
            model_name='examination',
            name='visit',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Visit'),
        ),
    ]
