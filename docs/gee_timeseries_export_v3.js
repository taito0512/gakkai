/**
 * 時系列特徴量エクスポート v3（EVI・NDMI・GLCMテクスチャ追加版）
 *
 * v2からの追加特徴量:
 *   【光学指数】
 *   - EVI_mean   : 年間EVI平均（NDVIより高NDVI域で識別力が高い）
 *   - NDMI_mean  : 年間NDMI平均（NIR-SWIR: 植生の水分含量を反映）
 *
 *   【SARテクスチャ（GLCM・年間平均VH画像から計算）】
 *   - VH_contrast : コントラスト（局所的な輝度変化の大きさ）
 *   - VH_ent      : エントロピー（テクスチャの複雑さ・不均一さ）
 *   - VH_idm      : 均質性 Inverse Difference Moment（滑らかさ）
 *   - VH_corr     : 相関（隣接ピクセル間の線形依存性）
 *
 * 使い方:
 *   1. REGION を切り替えて5地域分それぞれ実行
 *   2. Tasks タブから CSV をGoogle Driveにエクスポート
 *   3. ファイル名例: tsukubamirai_features_v3_2023.csv
 *
 * 処理時間の目安（v2比 約1.5〜2倍）:
 *   つくばみらい・稲敷・笠間・香取: 20〜30分
 *   十日町（ポリゴン数多・地形複雑）: 60〜90分
 */

// ============================================================
// ★ここを切り替えて5地域分実行する
// ============================================================
var REGION = 'tsukubamirai';
// 'tsukubamirai' / 'inashiki' / 'kasama' / 'katori' / 'tokamachi'

// ============================================================
// 設定
// ============================================================
var CONFIGS = {
  tsukubamirai: {
    shp:    'projects/my-project-taito/assets/2023_tukubamirai_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  inashiki: {
    shp:    'projects/my-project-taito/assets/2022_inashiki_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  kasama: {
    shp:    'projects/my-project-taito/assets/2022_kasama_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  katori: {
    shp:    'projects/my-project-taito/assets/2023_katori_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  tokamachi: {
    shp:    'projects/my-project-taito/assets/2023_tokamachi_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
};

var cfg  = CONFIGS[REGION];
var year = cfg.year;
var shp  = ee.FeatureCollection(cfg.shp);

// ============================================================
// Sentinel-2 年間統計（NDVI・NDWI・EVI・NDMI）
// ============================================================
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(shp.geometry())
  .filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(function(img) {
    // 各バンドをスケーリング（Sentinel-2 SR は 0〜10000）
    var b2  = img.select('B2').divide(10000);  // Blue
    var b4  = img.select('B4').divide(10000);  // Red
    var b8  = img.select('B8').divide(10000);  // NIR
    var b11 = img.select('B11').divide(10000); // SWIR1

    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');

    // EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    var evi = b8.subtract(b4)
      .multiply(2.5)
      .divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
      .rename('EVI');

    // NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    var ndmi = b8.subtract(b11).divide(b8.add(b11)).rename('NDMI');

    return img.addBands([ndvi, ndwi, evi, ndmi])
              .select(['NDVI', 'NDWI', 'EVI', 'NDMI'])
              .copyProperties(img, ['system:time_start']);
  });

// NDVI 統計量
var ndvi_col   = s2.select('NDVI');
var ndvi_mean  = ndvi_col.mean().rename('NDVI_mean');
var ndvi_max   = ndvi_col.max().rename('NDVI_max');
var ndvi_min   = ndvi_col.min().rename('NDVI_min');
var ndvi_std   = ndvi_col.reduce(ee.Reducer.stdDev()).rename('NDVI_std');
var ndvi_range = ndvi_max.subtract(ndvi_min).rename('NDVI_range');

// NDWI・EVI・NDMI 年間平均
var ndwi_mean = s2.select('NDWI').mean().rename('NDWI_mean');
var evi_mean  = s2.select('EVI').mean().rename('EVI_mean');
var ndmi_mean = s2.select('NDMI').mean().rename('NDMI_mean');

// ============================================================
// 月別NDVI中央値（作付けサイクルを捉える）
// ============================================================
function monthlyNDVI(month, name) {
  var start    = ee.Date.fromYMD(year, month, 1);
  var end      = start.advance(1, 'month');
  var med      = ndvi_col.filterDate(start, end).median();
  // 当月に雲なし画像がない場合は前後1ヶ月まで拡張
  var fallback = ndvi_col
    .filterDate(start.advance(-1, 'month'), end.advance(1, 'month'))
    .median();
  return ee.Image(ee.Algorithms.If(
    ndvi_col.filterDate(start, end).size().gt(0),
    med,
    fallback
  )).rename(name);
}

var ndvi_apr = monthlyNDVI(4,  'NDVI_apr');
var ndvi_jun = monthlyNDVI(6,  'NDVI_jun');
var ndvi_aug = monthlyNDVI(8,  'NDVI_aug');
var ndvi_oct = monthlyNDVI(10, 'NDVI_oct');

// ============================================================
// Sentinel-1 VH 年間統計
// ============================================================
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(shp.geometry())
  .filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select('VH');

var vh_mean = s1.mean().rename('VH_mean');
var vh_std  = s1.reduce(ee.Reducer.stdDev()).rename('VH_std');

// ============================================================
// GLCMテクスチャ（年間平均VHに3×3窓で計算）
// ============================================================
// VHはdB値（約-30〜0）→ 0〜255の整数にスケーリングしてGLCM計算
var vh_scaled = s1.mean()
  .add(30)               // -30〜0 → 0〜30
  .multiply(255.0 / 30)  // 0〜30 → 0〜255
  .toInt32()
  .clamp(0, 255)
  .rename('VH');

// glcmTexture はバンド名をプレフィックスに使う（例: VH_contrast）
var texture = vh_scaled.glcmTexture({ size: 3 });

var vh_contrast = texture.select('VH_contrast');
var vh_ent      = texture.select('VH_ent');   // エントロピー
var vh_idm      = texture.select('VH_idm');   // 均質性
var vh_corr     = texture.select('VH_corr');  // 相関

// ============================================================
// 全特徴量をまとめて筆ポリゴンごとに集計
// ============================================================
var feat_img = ee.Image.cat([
  // v2特徴量（互換性維持）
  ndvi_mean, ndvi_max, ndvi_min, ndvi_range, ndvi_std,
  ndwi_mean,
  ndvi_apr, ndvi_jun, ndvi_aug, ndvi_oct,
  vh_mean, vh_std,
  // v3追加: EVI・NDMI
  evi_mean, ndmi_mean,
  // v3追加: GLCMテクスチャ
  vh_contrast, vh_ent, vh_idm, vh_corr,
]);

var result = feat_img.reduceRegions({
  collection: shp,
  reducer:    ee.Reducer.mean(),
  scale:      10,
  tileScale:  8,  // テクスチャ計算はメモリ負荷が高いのでv2の4→8に増やす
});

// ============================================================
// エクスポート
// ============================================================
Export.table.toDrive({
  collection:     result,
  description:    REGION + '_features_v3_' + year,
  fileNamePrefix: REGION + '_features_v3_' + year,
  fileFormat:     'CSV',
  selectors: [
    cfg.id_col,
    // v2特徴量
    'NDVI_mean', 'NDVI_max', 'NDVI_min', 'NDVI_range', 'NDVI_std',
    'NDWI_mean',
    'NDVI_apr', 'NDVI_jun', 'NDVI_aug', 'NDVI_oct',
    'VH_mean', 'VH_std',
    // v3追加
    'EVI_mean', 'NDMI_mean',
    'VH_contrast', 'VH_ent', 'VH_idm', 'VH_corr',
  ],
});

// ============================================================
// デバッグ用出力
// ============================================================
print('対象地域:', REGION);
print('ポリゴン数:', shp.size());
print('Sentinel-2 シーン数:', s2.size());
print('Sentinel-1 シーン数:', s1.size());
print('4月シーン数:', ndvi_col.filterDate(year+'-04-01', year+'-05-01').size());
print('6月シーン数:', ndvi_col.filterDate(year+'-06-01', year+'-07-01').size());
print('8月シーン数:', ndvi_col.filterDate(year+'-08-01', year+'-09-01').size());
print('10月シーン数:', ndvi_col.filterDate(year+'-10-01', year+'-11-01').size());
