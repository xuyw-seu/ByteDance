# Dataset Details

Details of four network traffic datasets.

| Dataset | Classes |
|---------|---------|
| CSTNET-TLS1.3 | 163, 51.la, 51cto, acm, adobe, alibaba, alipay, amap, apple, arxiv, baidu, bilibili, biligame, booking, chia, cloudflare, cloudfront, cnblogs, criteo, deepl, eastday, facebook, feishu, ggph, github, gmail, google, huanqiu, huawei, ibm, icloud, ieee, jd, msn, netflix, nike, office, overleaf, qq, sciencedirect, snapchat, sohu, t.co, thepaper, weibo, wikimedia, xiaomi, yahoo, youtube, zhihu |
| MOBILE-APP | facebook, instagram, linkedIn, spotify, tiktok, twitter, wikipedia, youtube |
| TOR | 163, alibaba, aliyun, amazon, apple, baidu, bilibili, bing, cdc, csdn, deepl, epicgames, github, icloud, jd, msn, office, openai, qq, sci-encodedirect, stackoverflow, tencent, weibo, yahoo, zhihu |
| TROJAN-VPN | amazon, facebook, google, nytimes, reddit, twitter, wiki, youtube, weibo |



## Dataset Links

The following link points to the pre-processed version of the dataset provided by the authors:  
https://pan.baidu.com/s/1dFDEkobfCmwh6vbu9B_ifg?pwd=9m6t

## Dataset Description

This directory contains multiple preprocessed network traffic text files (`.txt`). Each file corresponds to an independent network session, organized in a unified three-column format, suitable for tasks such as traffic analysis, encrypted traffic identification, and time series modeling.

## Data Format Description

Each `.txt` file contains multiple records, with each line consisting of three columns separated by spaces or tabs:

1. **Relative Timestamp**
   Represents the time difference of the packet relative to the session start time.

2. **Packet Length with Direction**
   Direction is indicated by positive/negative signs:
   
   * Positive: Client → Server
   * Negative: Server → Client
   The value is the packet length in bytes.

3. **First-n Bytes Sequence**
   The first n bytes of the packet payload, represented as a byte sequence (can be decimal or hexadecimal). Used for byte feature analysis.

## Example

0.000123 64 23 45 6A 90 … 0.010532 -120 30 19 AF 01 … 0.015900 72 4F 3B 22 10 …

## Directory Contents

* Multiple `.txt` files, each representing an independent session.
* Filenames can be named based on session ID, five-tuple hash, or capture order.

## Notes

* Records within each file are ordered by time.



## Public Dataset Links

| Dataset | Download Link |
|---------|----------|
| CSTNET-TLS1.3 | [CSTNET-TLS1.3 数据集](https://drive.google.com/drive/folders/1BUo5TMRuXNvTqNYy0RLeHk4l4Q3BuzSk) |
| MOBILE-APP | [MOBILE-APP 数据集](https://ieee-dataport.org/documents/5g-mobile-app-traffic-traces-chalmers-2023) |
