"""
generate_mock_data.py
---------------------
Generates mock catalog.parquet and reviews.parquet files under data/
with the EXACT same column schema as the Amazon Reviews 2023 dataset.
Run this once before starting the Flask server if you don't have the real files.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Product catalog — 31 products across 8 categories
# ---------------------------------------------------------------------------
PRODUCTS = [
    # ── Water Bottles ──────────────────────────────────────────────────────
    {
        "parent_asin": "B001WAT001",
        "title": "HydroFlask 32oz Wide Mouth Water Bottle",
        "store": "HydroFlask",
        "price": 44.95,
        "average_rating": 4.8,
        "rating_number": 15234,
        "category_path": "Sports & Outdoors > Hydration > Water Bottles",
        "features_text": (
            "Double-wall vacuum insulation keeps beverages cold 24 hours, hot 12 hours. "
            "Food-grade 18/8 stainless steel, BPA-free. Wide mouth fits ice cubes. "
            "Powder-coated exterior for grip. 1-year manufacturer warranty against defects. "
            "30-day hassle-free return and exchange policy. Cash on delivery available."
        ),
        "description_text": (
            "The HydroFlask Wide Mouth bottle is the perfect companion for all adventures. "
            "Whether hiking, working out, or going through daily routines, it keeps beverages "
            "at the perfect temperature all day. Available in 20+ color options."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B001WAT002",
        "title": "YETI Rambler 26oz Vacuum Insulated Bottle",
        "store": "YETI",
        "price": 39.98,
        "average_rating": 4.7,
        "rating_number": 8921,
        "category_path": "Sports & Outdoors > Hydration > Water Bottles",
        "features_text": (
            "Dishwasher safe stainless steel. Double-wall vacuum insulation. "
            "No-sweat design. TripleHaul cap. Keeps cold 24+ hours, hot 6 hours. "
            "5-year limited warranty included. Cash on delivery available. "
            "60-day return and exchange policy accepted."
        ),
        "description_text": (
            "YETI Ramblers are built for the wild. Tough enough for all conditions, "
            "these insulated bottles keep drinks cold or hot all day long."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B001WAT003",
        "title": "Nalgene Tritan Wide Mouth Water Bottle 32oz",
        "store": "Nalgene",
        "price": 16.49,
        "average_rating": 4.6,
        "rating_number": 23456,
        "category_path": "Sports & Outdoors > Hydration > Water Bottles",
        "features_text": (
            "Made from BPA-free Tritan co-polyester. Wide mouth opening for easy filling. "
            "Dishwasher safe, top rack. Leak-proof loop-top lid. Lightweight at 6.2 oz. "
            "Lifetime guarantee against defects. No cash on delivery option. "
            "30-day return accepted in original condition."
        ),
        "description_text": (
            "The iconic Nalgene Tritan bottle is durable, lightweight, and completely free "
            "of BPA and BPA alternatives. Trusted by outdoor enthusiasts worldwide."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Smartphones ────────────────────────────────────────────────────────
    {
        "parent_asin": "B002PHN001",
        "title": "Samsung Galaxy S24 Ultra 256GB Titanium Black",
        "store": "Samsung",
        "price": 1199.99,
        "average_rating": 4.6,
        "rating_number": 5678,
        "category_path": "Electronics > Cell Phones > Smartphones",
        "features_text": (
            "6.8-inch Dynamic AMOLED 2X display with 120Hz. 200MP quad-camera system. "
            "Snapdragon 8 Gen 3 processor. 5000mAh battery with 45W fast charging. "
            "S Pen included. IP68 water resistance. "
            "1-year manufacturer warranty. Cash on delivery available in select cities. "
            "15-day return policy, device must be unopened."
        ),
        "description_text": (
            "The Galaxy S24 Ultra is the most powerful Galaxy device ever made. "
            "With an advanced AI-powered camera and built-in S Pen, it redefines productivity."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B002PHN002",
        "title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
        "store": "Apple",
        "price": 1199.00,
        "average_rating": 4.7,
        "rating_number": 7890,
        "category_path": "Electronics > Cell Phones > Smartphones",
        "features_text": (
            "6.7-inch Super Retina XDR ProMotion display. A17 Pro chip with 6-core GPU. "
            "48MP main camera with 5x Telephoto. Grade-5 titanium frame. USB-C with USB 3. "
            "Action button customizable. 1-year Apple warranty. No cash on delivery. "
            "14-day return policy with original packaging required."
        ),
        "description_text": (
            "iPhone 15 Pro Max brings a strong and light titanium design with the most "
            "advanced iPhone camera system ever — all powered by the A17 Pro chip."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B002PHN003",
        "title": "Google Pixel 8 Pro 128GB Obsidian",
        "store": "Google",
        "price": 799.00,
        "average_rating": 4.5,
        "rating_number": 3456,
        "category_path": "Electronics > Cell Phones > Smartphones",
        "features_text": (
            "6.7-inch LTPO OLED display 120Hz. 50MP main camera with Google AI. "
            "Tensor G3 chip. Temperature sensor. 5050mAh battery. "
            "7 years of OS and security updates guaranteed. "
            "1-year warranty. Cash on delivery not available. "
            "30-day return policy, full refund guaranteed."
        ),
        "description_text": (
            "Pixel 8 Pro is the most advanced Pixel ever built with the Google Tensor G3 chip "
            "and a pro-level camera with Google AI for extraordinary photos day and night."
        ),
        "n_reviews_sampled": 4,
    },
    # ── Over-Ear Headphones ────────────────────────────────────────────────
    {
        "parent_asin": "B003HPH001",
        "title": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        "store": "Sony",
        "price": 278.00,
        "average_rating": 4.8,
        "rating_number": 12345,
        "category_path": "Electronics > Headphones > Over-Ear > Noise Canceling",
        "features_text": (
            "Industry-leading noise cancellation with two processors and 8 microphones. "
            "30-hour battery life with quick charge (3 min = 3 hours). "
            "Multi-point connection to 2 devices simultaneously. LDAC Hi-Res Audio. "
            "Speak-to-chat auto pause. 1-year Sony warranty. "
            "Cash on delivery available. 30-day return and exchange."
        ),
        "description_text": (
            "Experience next-level noise canceling and extraordinary sound quality. "
            "The WH-1000XM5 features a new lightweight design built for all-day comfort."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B003HPH002",
        "title": "Bose QuietComfort 45 Bluetooth Noise Canceling Headphones",
        "store": "Bose",
        "price": 279.00,
        "average_rating": 4.5,
        "rating_number": 6543,
        "category_path": "Electronics > Headphones > Over-Ear > Noise Canceling",
        "features_text": (
            "Quiet mode (noise canceling) and Aware mode (pass-through). "
            "24-hour battery life per charge. Foldable, lightweight design. "
            "TriPort acoustic architecture for deep, immersive audio. "
            "USB-C charging. 1-year Bose warranty. "
            "Cash on delivery available in major cities. 30-day easy returns."
        ),
        "description_text": (
            "The QuietComfort 45 delivers everything you love about Bose noise canceling "
            "with high-fidelity audio and a refined, lightweight design."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B003HPH003",
        "title": "JBL Tune 760NC Wireless Noise Canceling Headphones",
        "store": "JBL",
        "price": 79.95,
        "average_rating": 4.3,
        "rating_number": 4321,
        "category_path": "Electronics > Headphones > Over-Ear > Noise Canceling",
        "features_text": (
            "Active Noise Cancelling technology. 35-hour battery life with ANC on. "
            "Foldable lightweight design. JBL Pure Bass Sound. "
            "Multi-point connection to 2 devices. USB-C charging. "
            "1-year warranty. Cash on delivery available. 30-day return."
        ),
        "description_text": (
            "JBL Tune 760NC headphones deliver exceptional 35-hour battery life with "
            "active noise cancellation powered by the beloved JBL Pure Bass sound."
        ),
        "n_reviews_sampled": 3,
    },
    # ── True Wireless Earbuds ──────────────────────────────────────────────
    {
        "parent_asin": "B003EAR001",
        "title": "Apple AirPods Pro (2nd Generation) with MagSafe Case",
        "store": "Apple",
        "price": 199.00,
        "average_rating": 4.6,
        "rating_number": 9876,
        "category_path": "Electronics > Headphones > True Wireless Earbuds",
        "features_text": (
            "Active Noise Cancellation, 2x better than 1st gen. Adaptive Transparency. "
            "Personalized Spatial Audio with dynamic head tracking. H2 chip. "
            "Up to 6 hours listening time (30 hours with case). IPX4 water resistant. "
            "1-year Apple warranty. 14-day return policy. No cash on delivery."
        ),
        "description_text": (
            "AirPods Pro (2nd gen) are the world's best-selling earbuds with breakthrough "
            "noise cancellation, Adaptive Audio, and Personalized Spatial Audio."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B003EAR002",
        "title": "Samsung Galaxy Buds2 Pro True Wireless Earbuds",
        "store": "Samsung",
        "price": 149.99,
        "average_rating": 4.4,
        "rating_number": 5678,
        "category_path": "Electronics > Headphones > True Wireless Earbuds",
        "features_text": (
            "Intelligent ANC with 3 microphones. 360 Audio surround sound. "
            "IPX7 water resistance. Up to 8 hours playtime (29 total with case). "
            "Hi-Fi 24-bit audio. Bixby and voice assistant support. "
            "1-year Samsung warranty. Cash on delivery available. 15-day return."
        ),
        "description_text": (
            "Galaxy Buds2 Pro deliver Hi-Fi 24-bit sound quality with Intelligent ANC "
            "for a truly immersive listening experience in an ultra-compact design."
        ),
        "n_reviews_sampled": 3,
    },
    {
        "parent_asin": "B003EAR003",
        "title": "Jabra Elite 4 Active True Wireless Sport Earbuds",
        "store": "Jabra",
        "price": 79.99,
        "average_rating": 4.3,
        "rating_number": 3456,
        "category_path": "Electronics > Headphones > True Wireless Earbuds",
        "features_text": (
            "Active Noise Cancellation. HearThrough mode. IP57 water and dustproof. "
            "7 hours battery (28 total with case). Multipoint connection to 2 devices. "
            "Secure sport fit with ShakeGrip. 2-year warranty. "
            "30-day return. Cash on delivery available."
        ),
        "description_text": (
            "Jabra Elite 4 Active are designed for sport with IP57 rating and ANC. "
            "The secure ShakeGrip coating keeps them in place during the most intense workouts."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Laptops ────────────────────────────────────────────────────────────
    {
        "parent_asin": "B004LAP001",
        "title": "Apple MacBook Pro 14-inch M3 Pro 18GB 512GB",
        "store": "Apple",
        "price": 1999.00,
        "average_rating": 4.8,
        "rating_number": 2345,
        "category_path": "Electronics > Computers > Laptops > MacBooks",
        "features_text": (
            "Apple M3 Pro chip with 11-core CPU and 14-core GPU. 18GB unified memory. "
            "512GB SSD. 14.2-inch Liquid Retina XDR display 3024x1964. "
            "Up to 18 hours battery life. Three Thunderbolt 4 ports. HDMI, SD card slot. "
            "1-year AppleCare warranty. No cash on delivery. 14-day return policy."
        ),
        "description_text": (
            "MacBook Pro blows the doors off what a pro laptop can do. The breathtakingly fast "
            "M3 Pro chip, stunning Liquid Retina XDR display, and all-day battery make it "
            "the ultimate professional laptop."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B004LAP002",
        "title": "Dell XPS 15 9530 Core i7 RTX 4060 OLED 512GB",
        "store": "Dell",
        "price": 1499.99,
        "average_rating": 4.4,
        "rating_number": 1876,
        "category_path": "Electronics > Computers > Laptops > Windows",
        "features_text": (
            "Intel Core i7-13700H processor. 16GB DDR5 RAM. 512GB NVMe SSD. "
            "NVIDIA GeForce RTX 4060 8GB. 15.6-inch OLED 3.5K 60Hz display. "
            "86Wh battery, up to 13 hours. Thunderbolt 4, USB-A, SD card. "
            "1-year Dell ProSupport warranty. Cash on delivery available. "
            "30-day return and exchange accepted."
        ),
        "description_text": (
            "The Dell XPS 15 combines elegant, premium design with powerful performance. "
            "The stunning OLED display and NVIDIA RTX 4060 make it perfect for creative work."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B004LAP003",
        "title": "Lenovo ThinkPad X1 Carbon Gen 11 Core i7 16GB",
        "store": "Lenovo",
        "price": 1349.00,
        "average_rating": 4.6,
        "rating_number": 1543,
        "category_path": "Electronics > Computers > Laptops > Business",
        "features_text": (
            "Intel Core i7-1365U vPro. 16GB LPDDR5 RAM. 512GB SSD. "
            "14-inch IPS anti-glare display. Carbon fiber chassis under 1.12kg. "
            "Military-grade MIL-SPEC durability (MIL-STD-810H). "
            "3-year warranty with on-site support included. No cash on delivery. "
            "30-day business return policy."
        ),
        "description_text": (
            "The ThinkPad X1 Carbon Gen 11 is the world's lightest 14-inch business laptop, "
            "engineered for professionals who demand performance, reliability, and security."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Tablets ────────────────────────────────────────────────────────────
    {
        "parent_asin": "B005TAB001",
        "title": "Apple iPad Air 5th Generation 64GB WiFi Blue",
        "store": "Apple",
        "price": 549.00,
        "average_rating": 4.7,
        "rating_number": 4321,
        "category_path": "Electronics > Tablets > iPad",
        "features_text": (
            "Apple M1 chip for desktop-class performance. 10.9-inch Liquid Retina display. "
            "5G capable (WiFi + Cellular model available). USB-C with 2x faster charging. "
            "Works with Apple Pencil (2nd gen) and Magic Keyboard. "
            "1-year Apple warranty. No cash on delivery. 14-day return."
        ),
        "description_text": (
            "iPad Air with M1 is a powerhouse thin and light tablet perfect for creative work, "
            "studying, gaming, and entertainment. Available in five gorgeous colors."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B005TAB002",
        "title": "Samsung Galaxy Tab S9 256GB WiFi Graphite",
        "store": "Samsung",
        "price": 649.99,
        "average_rating": 4.6,
        "rating_number": 3210,
        "category_path": "Electronics > Tablets > Android",
        "features_text": (
            "Qualcomm Snapdragon 8 Gen 2 processor. 11-inch Dynamic AMOLED 2X display 120Hz. "
            "IP68 water and dust resistance. S Pen included with storage. 12GB RAM. "
            "8,400mAh battery. 45W super fast charging. "
            "1-year Samsung warranty. Cash on delivery available. 30-day return."
        ),
        "description_text": (
            "Galaxy Tab S9 redefines what a tablet can do with its premium build, "
            "stunning Dynamic AMOLED display, and powerful Snapdragon 8 Gen 2 chip."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Smartwatches ───────────────────────────────────────────────────────
    {
        "parent_asin": "B006SMA001",
        "title": "Apple Watch Series 9 GPS 45mm Midnight Aluminum",
        "store": "Apple",
        "price": 429.00,
        "average_rating": 4.7,
        "rating_number": 6789,
        "category_path": "Electronics > Wearables > Smartwatches",
        "features_text": (
            "Apple S9 chip with Double Tap gesture. Always-on Retina display. "
            "ECG app, Blood Oxygen sensor, Crash Detection, and Fall Detection. "
            "18-hour battery life. WR50 water resistance (50 meters). "
            "1-year Apple warranty. No cash on delivery. 14-day return policy."
        ),
        "description_text": (
            "Apple Watch Series 9 is a powerful health and wellness companion with "
            "advanced health features, a stunning always-on display, and the innovative Double Tap gesture."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B006SMA002",
        "title": "Samsung Galaxy Watch6 Classic 47mm Black",
        "store": "Samsung",
        "price": 299.99,
        "average_rating": 4.5,
        "rating_number": 4567,
        "category_path": "Electronics > Wearables > Smartwatches",
        "features_text": (
            "BioActive Sensor for health monitoring. Advanced sleep coaching. "
            "5ATM + IP68 water resistance. Classic rotating bezel. "
            "40-hour battery life. 1.5-inch Super AMOLED display. "
            "1-year Samsung warranty. Cash on delivery available. "
            "15-day return and exchange policy."
        ),
        "description_text": (
            "Galaxy Watch6 Classic brings back the iconic rotating bezel with advanced "
            "health tracking and a premium design for the modern professional."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Books ──────────────────────────────────────────────────────────────
    {
        "parent_asin": "B007BOK001",
        "title": "Atomic Habits by James Clear — Hardcover",
        "store": "Avery",
        "price": 15.99,
        "average_rating": 4.9,
        "rating_number": 45678,
        "category_path": "Books > Self-Help > Personal Development",
        "features_text": (
            "Hardcover, 320 pages. #1 New York Times bestseller. "
            "Practical strategies for building good habits and breaking bad ones. "
            "No warranty applicable to books. 30-day return if unopened and in original condition. "
            "Cash on delivery available."
        ),
        "description_text": (
            "No matter your goals, Atomic Habits offers a proven framework for improving "
            "1% every day. James Clear reveals simple, practical strategies that can "
            "transform your habits and deliver remarkable results."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B007BOK002",
        "title": "Deep Work by Cal Newport — Paperback",
        "store": "Grand Central Publishing",
        "price": 14.99,
        "average_rating": 4.7,
        "rating_number": 23456,
        "category_path": "Books > Business > Productivity",
        "features_text": (
            "Paperback edition, 304 pages. Rules for focused success in a distracted world. "
            "No warranty on books. 30-day return if in original unread condition. "
            "Cash on delivery available."
        ),
        "description_text": (
            "Deep Work is an indispensable guide for anyone seeking focused success in a "
            "distracted world. Cal Newport argues that the ability to perform deep work "
            "is becoming increasingly rare and valuable."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B007BOK003",
        "title": "The Psychology of Money by Morgan Housel — Hardcover",
        "store": "Harriman House",
        "price": 18.00,
        "average_rating": 4.8,
        "rating_number": 34567,
        "category_path": "Books > Business > Finance",
        "features_text": (
            "Hardcover, 256 pages. Timeless lessons on wealth, greed, and happiness. "
            "Wall Street Journal bestseller. No warranty. 30-day return policy. "
            "Cash on delivery available."
        ),
        "description_text": (
            "Morgan Housel shares 19 short stories exploring the strange ways people "
            "think about money and teaches you how to make better sense of one of life's most important topics."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B007BOK004",
        "title": "Thinking, Fast and Slow by Daniel Kahneman — Paperback",
        "store": "Farrar, Straus and Giroux",
        "price": 17.00,
        "average_rating": 4.6,
        "rating_number": 28901,
        "category_path": "Books > Psychology > Cognitive Science",
        "features_text": (
            "Paperback, 499 pages. By Nobel Prize laureate in Economics. "
            "Groundbreaking work on behavioral economics and human decision-making. "
            "No warranty applicable. 30-day return if unused. COD available."
        ),
        "description_text": (
            "Nobel laureate Daniel Kahneman's work on the two systems that drive human "
            "thinking — System 1 (fast, intuitive) and System 2 (slow, deliberate)."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Clothing ───────────────────────────────────────────────────────────
    {
        "parent_asin": "B008CLO001",
        "title": "Levi's 511 Slim Fit Jeans Men's",
        "store": "Levi's",
        "price": 59.50,
        "average_rating": 4.5,
        "rating_number": 18765,
        "category_path": "Clothing > Men's > Pants > Jeans",
        "features_text": (
            "99% cotton, 1% elastane for slight stretch. Slim fit through hip and thigh. "
            "Machine washable, cold water. Available in multiple washes and sizes. "
            "30-day exchange for wrong size or defect. Cash on delivery available."
        ),
        "description_text": (
            "Levi's 511 slim fit jeans sit below the waist with a slim leg from hip to ankle. "
            "The slight stretch makes these jeans perfect for all-day comfort and style."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B008CLO002",
        "title": "Nike Air Max 270 Men's Running Shoes",
        "store": "Nike",
        "price": 150.00,
        "average_rating": 4.6,
        "rating_number": 12890,
        "category_path": "Clothing > Men's > Shoes > Athletic",
        "features_text": (
            "Mesh upper for superior breathability. Max Air unit in heel for comfort. "
            "Foam midsole for lightweight cushioning. Rubber outsole for traction. "
            "2-year warranty against manufacturing defects. "
            "60-day return policy for athletic footwear. Cash on delivery available."
        ),
        "description_text": (
            "Nike Air Max 270 delivers visible cushioning under every step. "
            "Updated from OG running shoes of the late 80s with a modern lifestyle look."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B008CLO003",
        "title": "Adidas Ultraboost 22 Men's Running Shoes",
        "store": "Adidas",
        "price": 139.95,
        "average_rating": 4.5,
        "rating_number": 9876,
        "category_path": "Clothing > Men's > Shoes > Running",
        "features_text": (
            "Primeknit+ adaptive upper hugs the foot. BOOST midsole for energy return. "
            "Continental rubber outsole for grip. Torsion System for stability. "
            "2-year warranty on manufacturing defects. 30-day return. "
            "Cash on delivery available."
        ),
        "description_text": (
            "Adidas Ultraboost 22 delivers extraordinary cushioning and energy return. "
            "Designed for runners who love their craft with the iconic BOOST technology."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B008CLO004",
        "title": "Columbia Men's Watertight II Packable Rain Jacket",
        "store": "Columbia",
        "price": 79.95,
        "average_rating": 4.5,
        "rating_number": 11234,
        "category_path": "Clothing > Men's > Jackets > Rain Jackets",
        "features_text": (
            "Waterproof seam-sealed shell with Omni-Tech breathability. "
            "Packable into its own pocket. Adjustable hood and cuffs. "
            "Multiple color options. 1-year warranty on manufacturing defects. "
            "30-day return and exchange. Cash on delivery available."
        ),
        "description_text": (
            "Columbia Watertight II is your go-to rain jacket for all outdoor adventures. "
            "Fully seam-sealed and waterproof, it packs away compactly for easy travel."
        ),
        "n_reviews_sampled": 3,
    },
    # ── Kitchen Appliances ─────────────────────────────────────────────────
    {
        "parent_asin": "B009KIT001",
        "title": "Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt",
        "store": "Instant Pot",
        "price": 89.99,
        "average_rating": 4.7,
        "rating_number": 34567,
        "category_path": "Home & Kitchen > Appliances > Pressure Cookers",
        "features_text": (
            "7-in-1 functions: pressure cooker, slow cooker, rice cooker, steamer, "
            "saute pan, yogurt maker, and food warmer. UL and ULC certified. "
            "1-year warranty included. 30-day return policy. Cash on delivery available."
        ),
        "description_text": (
            "Instant Pot Duo is America's #1 best-selling multi-cooker. "
            "Replaces up to 7 kitchen appliances. Cooks meals up to 70% faster."
        ),
        "n_reviews_sampled": 5,
    },
    {
        "parent_asin": "B009KIT002",
        "title": "Ninja BL610 Professional Countertop Blender 1000W",
        "store": "Ninja",
        "price": 89.99,
        "average_rating": 4.6,
        "rating_number": 23456,
        "category_path": "Home & Kitchen > Appliances > Blenders",
        "features_text": (
            "1000-watt motor for powerful blending. 72 oz Total Crushing pitcher. "
            "BPA-free pitcher. Dishwasher-safe parts (except motor base). "
            "1-year warranty included. Cash on delivery available. 30-day return policy."
        ),
        "description_text": (
            "Ninja Professional Blender crushes ice, blends ingredients, makes smoothies, "
            "and frozen drinks with ease using its powerful 1000-watt motor."
        ),
        "n_reviews_sampled": 4,
    },
    {
        "parent_asin": "B009KIT003",
        "title": "Keurig K-Classic Coffee Maker Single-Serve K-Cup Pod",
        "store": "Keurig",
        "price": 99.99,
        "average_rating": 4.5,
        "rating_number": 45678,
        "category_path": "Home & Kitchen > Appliances > Coffee Makers",
        "features_text": (
            "Brews 6, 8, or 10 oz cup sizes. 48 oz removable water reservoir. "
            "Compatible with all K-Cup pods. Auto-off feature. Descaling reminder. "
            "1-year warranty. 30-day return. Cash on delivery available in select areas."
        ),
        "description_text": (
            "Keurig K-Classic brews the perfect cup of coffee in under one minute. "
            "Simple button controls and a large reservoir make it ideal for home use."
        ),
        "n_reviews_sampled": 4,
    },
]

# ---------------------------------------------------------------------------
# Review templates per product category
# ---------------------------------------------------------------------------
REVIEW_TEMPLATES = {
    "Water Bottles": [
        ("Keeps water cold all day!", "I've had this bottle for 6 months and it still keeps my water ice cold for over 24 hours. Absolutely worth every penny.", 5, 45),
        ("Great quality, slight dent issue", "The insulation is fantastic and I love the wide mouth design. However, mine arrived with a small dent. Customer service was helpful.", 4, 12),
        ("Perfect for hiking", "Took this on a 3-day hiking trip and it performed flawlessly. The powder coat feels premium and the lid never leaked.", 5, 89),
        ("Good but pricey", "Does exactly what it says — keeps drinks cold. My only complaint is the price tag for what is essentially a metal bottle.", 4, 7),
        ("Best water bottle I've owned", "I've tried many brands and this is by far the best. The temperature retention is unmatched and it looks gorgeous.", 5, 134),
    ],
    "Smartphones": [
        ("Incredible camera!", "The camera on this phone is absolutely phenomenal. Night mode photos look like they were taken with a DSLR. Battery also lasts all day.", 5, 203),
        ("Fast and smooth performance", "Everything about this phone screams premium. Apps open instantly, gaming is butter smooth, and the display is stunning.", 5, 167),
        ("Great phone, minor issues", "Love the phone overall but the charging speed could be faster and the camera app sometimes crashes. Software updates fixed most issues.", 4, 45),
        ("Battery could be better", "Performance and camera are top-notch, but I'm a heavy user and need to charge by 7pm. Otherwise a fantastic device.", 4, 88),
        ("Absolutely love it!", "Switched from my old phone and this is a massive upgrade. The display colors are vivid, camera is sharp, and it feels premium.", 5, 312),
    ],
    "Headphones": [
        ("Best noise canceling ever!", "I work in a noisy open office and these headphones are life-changing. The noise canceling is absolutely phenomenal.", 5, 456),
        ("Incredible sound quality", "The audio is rich, detailed, and has great bass without being overwhelming. Wear these all day and my ears don't hurt.", 5, 234),
        ("Comfortable for long sessions", "Wore these for an 8-hour work day and had zero discomfort. The cushioning is perfect and they're surprisingly lightweight.", 5, 178),
        ("Good but not perfect", "Sound quality is great and ANC works well. The app could be more intuitive and multipoint connection sometimes drops.", 4, 56),
        ("Worth every penny", "I was skeptical about the price but these are genuinely the best headphones I've ever owned. Noise canceling is next level.", 5, 389),
    ],
    "Earbuds": [
        ("Best earbuds I've used!", "The ANC is surprisingly effective for such small earbuds. Sound is clear and balanced, and the fit is very secure.", 5, 234),
        ("Great for workouts", "IPX rating is no joke — wore these in heavy rain and they survived. Sound quality is excellent for the price.", 5, 167),
        ("Connection is rock solid", "I've had issues with Bluetooth dropouts before but these stay connected perfectly even with my phone across the room.", 5, 89),
        ("Battery life is impressive", "Getting 7+ hours on a single charge consistently. Case charges them quickly too. Only wish they came in more colors.", 4, 45),
        ("Comfortable fit", "Found the right ear tip size and these disappear in my ears. ANC is effective on trains and planes.", 4, 112),
    ],
    "Laptops": [
        ("Blazing fast performance!", "This laptop handles everything I throw at it — video editing, coding, gaming. The display colors are absolutely stunning.", 5, 178),
        ("Best laptop I've ever owned", "Made the switch and couldn't be happier. Build quality is exceptional, keyboard is perfect, and battery lasts all day.", 5, 234),
        ("Great for programming", "Runs my development environment flawlessly. The display is sharp, keyboard is comfortable, and I'm still getting 10+ hours battery.", 5, 145),
        ("Premium but worth it", "Yes it's expensive, but the build quality, display, and performance justify every dollar. This will last 5+ years.", 5, 89),
        ("Minor display issue", "Performance is exceptional and I love the keyboard. Had a slight backlight bleed issue that was resolved under warranty quickly.", 4, 67),
    ],
    "Tablets": [
        ("Perfect for drawing and note-taking", "Paired with the stylus, this tablet has replaced my notebook completely. The display is gorgeous and latency is minimal.", 5, 123),
        ("Great for students", "Use it for classes, note-taking, and watching lectures. Battery easily lasts a full school day and it's lightweight.", 5, 89),
        ("Fast and capable", "Handles everything from video editing to gaming without breaking a sweat. The display is one of the best I've seen.", 5, 156),
        ("Good tablet, pricey accessories", "The tablet itself is excellent but buying the stylus and keyboard separately adds significantly to the cost.", 4, 78),
        ("Excellent media consumption device", "Display is stunning for movies and shows. Audio from the speakers is surprisingly good for a tablet.", 5, 201),
    ],
    "Smartwatches": [
        ("Excellent health tracking!", "The sleep tracking has genuinely improved my sleep habits. Heart rate monitoring is accurate compared to my gym equipment.", 5, 167),
        ("Best smartwatch features", "Does everything I need — fitness tracking, notifications, payments. Battery lasts 2 days of heavy use.", 5, 234),
        ("Great design, average battery", "Love the look and the health features are comprehensive. Battery life could be better — I charge every 1.5 days.", 4, 89),
        ("Accurate fitness tracking", "Compared my step count to a dedicated fitness tracker and it's impressively accurate. GPS works well outdoors.", 5, 145),
        ("Seamless phone integration", "Notifications, calls, and apps work perfectly with my phone. Setup was straightforward and took under 5 minutes.", 5, 112),
    ],
    "Books": [
        ("Life-changing book!", "Finished this in two days because I couldn't put it down. Already implementing the strategies and seeing real results.", 5, 567),
        ("Must-read for everyone", "Whether you're a student, professional, or retiree, this book has something valuable for you. Highly recommend.", 5, 445),
        ("Practical and actionable", "Not just theory — packed with concrete examples and practical advice you can apply immediately.", 5, 389),
        ("Good but dense", "Takes time to get through all the concepts but it's worth it. Some chapters are more impactful than others.", 4, 89),
        ("Changed my perspective", "I've read many books in this genre but this one actually shifted how I think and act. Remarkable writing.", 5, 678),
    ],
    "Clothing": [
        ("Perfect fit!", "Ordered my usual size and the fit was perfect. Quality feels premium and the material is comfortable all day.", 5, 145),
        ("Great quality for the price", "Durable stitching, comfortable material, and looks exactly like the photos. Very happy with this purchase.", 5, 234),
        ("Runs slightly small", "Quality is excellent but I'd recommend sizing up. Once I exchanged for the right size, it was perfect.", 4, 89),
        ("Comfortable and stylish", "Worn this multiple times and it still looks new after washing. Great color options too.", 5, 178),
        ("Good but shipping was slow", "Product quality is excellent once it arrived. Took longer than expected to ship but worth the wait.", 4, 56),
    ],
    "Appliances": [
        ("Changed my kitchen routine!", "I use this every single day. Saves so much time and the results are consistently excellent.", 5, 456),
        ("Works exactly as advertised", "Easy to set up, intuitive to use, and produces great results. Cleanup is also straightforward.", 5, 345),
        ("Great value appliance", "Does what it says and does it well. Build quality feels solid and I expect it to last years.", 5, 234),
        ("Minor learning curve", "Took a bit to figure out all the settings but once I did, it became indispensable in my kitchen.", 4, 89),
        ("Worth the investment", "Was hesitant given the price but it has genuinely made cooking faster and more enjoyable.", 5, 312),
    ],
}

def get_category_group(category_path: str) -> str:
    """Map category path to a template group."""
    cp = category_path.lower()
    if "water bottle" in cp or "hydration" in cp:
        return "Water Bottles"
    if "smartphone" in cp or "cell phone" in cp:
        return "Smartphones"
    if "over-ear" in cp:
        return "Headphones"
    if "true wireless" in cp or "earbud" in cp:
        return "Earbuds"
    if "laptop" in cp or "macbook" in cp:
        return "Laptops"
    if "tablet" in cp or "ipad" in cp:
        return "Tablets"
    if "smartwatch" in cp or "wearable" in cp:
        return "Smartwatches"
    if "book" in cp:
        return "Books"
    if "clothing" in cp or "shoe" in cp or "jacket" in cp or "jean" in cp:
        return "Clothing"
    if "appliance" in cp or "kitchen" in cp or "cooker" in cp or "blender" in cp or "coffee" in cp:
        return "Appliances"
    return "Appliances"


def generate_reviews(products: list[dict]) -> list[dict]:
    reviews = []
    base_date = datetime(2023, 1, 1)

    for product in products:
        asin = product["parent_asin"]
        n = product["n_reviews_sampled"]
        group = get_category_group(product["category_path"])
        templates = REVIEW_TEMPLATES.get(group, REVIEW_TEMPLATES["Appliances"])

        avg_rating = product["average_rating"]
        sampled = random.sample(templates, min(n, len(templates)))

        for i, (title, text, base_stars, votes) in enumerate(sampled):
            # Jitter rating near product average
            rating = min(5, max(1, round(avg_rating + random.uniform(-0.5, 0.5))))
            reviews.append({
                "parent_asin": asin,
                "rating": float(rating),
                "title": title,
                "text": text,
                "helpful_vote": votes + random.randint(-5, 15),
                "verified_purchase": random.random() > 0.2,
                "timestamp": int((base_date + timedelta(days=random.randint(0, 700))).timestamp() * 1000),
            })

    return reviews


def main():
    print(f"Generating {len(PRODUCTS)} products...")
    catalog_df = pd.DataFrame(PRODUCTS)

    # Ensure correct dtypes
    catalog_df["price"] = catalog_df["price"].astype(float)
    catalog_df["average_rating"] = catalog_df["average_rating"].astype(float)
    catalog_df["rating_number"] = catalog_df["rating_number"].astype(int)
    catalog_df["n_reviews_sampled"] = catalog_df["n_reviews_sampled"].astype(int)

    catalog_path = os.path.join(DATA_DIR, "catalog.parquet")
    catalog_df.to_parquet(catalog_path, index=False)
    print(f"  ✓ Saved catalog to {catalog_path} ({len(catalog_df)} rows)")

    print("Generating reviews...")
    reviews_data = generate_reviews(PRODUCTS)
    reviews_df = pd.DataFrame(reviews_data)
    reviews_df["rating"] = reviews_df["rating"].astype(float)
    reviews_df["helpful_vote"] = reviews_df["helpful_vote"].astype(int)
    reviews_df["verified_purchase"] = reviews_df["verified_purchase"].astype(bool)
    reviews_df["timestamp"] = reviews_df["timestamp"].astype(int)

    reviews_path = os.path.join(DATA_DIR, "reviews.parquet")
    reviews_df.to_parquet(reviews_path, index=False)
    print(f"  ✓ Saved reviews to {reviews_path} ({len(reviews_df)} rows)")

    print("\nColumn verification:")
    print(f"  catalog columns : {list(catalog_df.columns)}")
    print(f"  reviews columns : {list(reviews_df.columns)}")
    print("\nAll done! You can now start the Flask server.")


if __name__ == "__main__":
    main()
