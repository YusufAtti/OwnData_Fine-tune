from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, pipeline
import pandas as pd
from datasets import Dataset
import torch

torch.cuda.empty_cache()

# Veri seti yüklemesi
df = pd.read_csv('engosmanli400csv.csv')  # Veri setinizin yolunu buraya ekleyin

# CSV dosyanızın sütun adlarını kontrol edin, çıkıtımızdaki ilk 4 yazımızı bu kısım oluşturuyor
print(df.head())


# Tokenizer ve model yüklemesi
model_name = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# pad_token ayarlama
# Bu satır, padding token'ını ayarlar.
# Padding token'ı, değişken uzunluktaki metinlerin
# sabit uzunluğa getirilmesi için kullanılır.------
# GPT-2 modelinde genellikle eos_token (end-of-sentence)
# aynı zamanda padding token'ı olarak kullanılır.
tokenizer.pad_token = tokenizer.eos_token

# Pandas DataFrame'i Dataset'e dönüştürme
# Bu satır, Pandas DataFrame'inizi bir Dataset objesine dönüştürür.
# Dataset objesi, Trainer ile daha kolay çalışmayı sağlar.
dataset = Dataset.from_pandas(df)


# Tokenize işlevi tanımlama
# Bu kod bloğu, tokenize_function adlı bir fonksiyon tanımlar.
# Bu fonksiyon, her bir veri örneğini modele beslenmeden önce tokenlara dönüştürür.
def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, padding='max_length', max_length=128)


# Veri setini tokenize etme
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator tanımlama
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Eğitim parametreleri ayarlama


#BU KISIMLARDAKİ RAKAMSAL DEĞİŞİKLİK DOĞRULUK ORANINI NASIL ETKİLER?
training_args = TrainingArguments(

    #Bu satır, modelinizin çıktı dizinini (output directory) './results' klasörü olarak ayarlar. Çıktı dizini, modelinizin eğitim süreci boyunca ürettiği dosyaların (checkpoint dosyaları, kayıt dosyaları, modeller vb.) saklandığı yerdir.
    #Bu dizinin önceden var olması gerekir. Aksi takdirde, kodunuz bir hata verebilir.
    output_dir="./results",
    #Bu satır, çıktı dizisini üzerine yazma (overwrite output directory) özelliğini etkinleştirir. Bu özellik, './results' dizininde zaten var olan dosyaları siler ve modelinizin yeni dosyalarını buraya yazar.
    overwrite_output_dir=True,
    #Bu satır, değerlendirme stratejisini (evaluation strategy) 'epoch' olarak ayarlar. Değerlendirme stratejisi, modelinizin ne sıklıkla değerlendirileceğini ve hangi metriklerin kullanılacağını belirler.
    #epoch: Modeliniz her eğitim döneminin sonunda değerlendirilecektir.
    #'step': Modeliniz her belirli sayıda adımda değerlendirilecektir.
    #'auto': Kütüphane, modeliniz için en uygun değerlendirme stratejisini otomatik olarak seçecektir.
    #Daha sık değerlendirme, daha fazla bilgi sağlar ancak eğitim sürecini yavaşlatabilir.
    eval_strategy="epoch",
    #öğrenme oranını temsil eder, bu oranı Kodunuzdaki "learning_rate=2e-5,"
    # satırı, modelinizin öğrenme oranını 0.00002 olarak ayarlıyor. Bu oldukça küçük bir değer ve modelinizin yavaş öğrenmesine neden olabilir.
    # Modelinizi eğitirken daha yüksek bir öğrenme oranı istiyorsanız, bu satırı şu şekilde değiştirebilirsiniz: learning_rate=0.01
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    #Bu satır, ağırlık çürümesi (weight decay) parametresini 0.01 olarak ayarlar. Ağırlık çürümesi, makine öğrenimi modellerinde kullanılan bir düzenleme (regularization) tekniğidir. Bu teknik, modelin parametrelerinin (ağırlıklarının) çok büyük değerler almasını önler ve bu sayede modelin genelleştirme performansını (yeni veriler üzerindeki performansı) iyileştirmeye yardımcı olur.
    weight_decay=0.01,
    #Bu satır, kayıt dizinini (logging directory) './logs' klasörü olarak ayarlar. Kayıt dizini, modelin eğitim süreci boyunca üretilen log bilgileri ve performans ölçümleri gibi verilerin saklandığı yerdir.
    logging_dir='./logs',
    #Bu satır, kayıt aralığını (logging step) 10 olarak ayarlar. Kayıt aralığı, modelin kaç adımda bir kayıt bilgilerini ve performans ölçümlerini kaydetmesini belirtir. 10 değeri, her 10 adımda bir kayıt yapılmasını sağlar.
    logging_steps=10,
    #Bu satır, kaydetme aralığını (save step) 10 olarak ayarlar. Kaydetme aralığı, modelin kaç adımda bir checkpoint (kontrol noktası) dosyaları kaydetmesini belirtir. 10 değeri, her 10 adımda bir checkpoint dosyası kaydedilmesini sağlar
    save_steps=10,
    #Bu satır, raporlama (reporting) özelliğini devre dışı bırakır. Raporlama özelliği, modelin eğitim süreci boyunca belirli istatistikleri ve performans ölçümlerini ekrana yazdırır. "none" değeri, bu bilgilerin ekrana yazdırılmasını engeller.
    report_to="none",

    #save_steps ve logging_steps farklı değerlerdeyse, checkpoint dosyalarıyla eşleşen eğitim istatistiklerini bulmak zor olabilir.

)

# Trainer tanımlama

#Transformers kütüphanesinden bir Trainer nesnesi oluşturur ve bu nesne, modelinizi eğitmek ve değerlendirmek için kullanılır.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
)


# Eğitimi başlatma
trainer.train()

# Model değerlendirmesi
#evaluate() fonksiyonu, modelinizi değerlendirme veri kümesi üzerinde çalıştırır ve kayıp, doğruluk gibi çeşitli metrikleri hesaplar. print(results) satırı, bu metrikleri ekrana yazdırır.
results = trainer.evaluate()
print(results)

# Fine-tuned modeli kaydetme
model_path = "./fine_tuned_gpt2_turkish"
#eğitimli modelinizi (fine-tuned model) belirtilen yola kaydeder. Bu model, yeni veriler üzerinde tahminler yapmak veya daha fazla geliştirmek için kullanılabilir.
trainer.save_model(model_path)

# Fine-tuned modeli kullanarak metin üretimi
fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_path)
text_generator = pipeline('text-generation', model=fine_tuned_model, tokenizer=tokenizer, device=0)

# Örnek metin üretimi
generated_text = text_generator("Who was the Ottoman Sultan during the Treaty of Jassy?", max_length=50, truncation=True)
print(generated_text)
