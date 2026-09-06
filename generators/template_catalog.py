# -*- coding: utf-8 -*-
"""
The built-in template catalogue.

This is data, not drawing code, and it is kept free of QGIS so that anything
can read it: the tests, the diagnostics report, and the subject matcher that
turns Korean artefact terms in a user's note into the English names an AI
model understands.

Each entry carries a ``default_color`` and a ``category``. Most also carry an
explicit ``draw`` entry naming the painter and its arguments; the rest are
matched by keyword in TemplateGenerator._resolve_draw. A ``file`` entry is
optional and names an SVG shipped alongside the plugin, which is used in place
of the code drawing when the file is actually present.

Names are English identifiers: they key this dictionary, get stored in
settings and travel with saved projects. Only the label shown in the UI is
translated (see i18n_ko.py).
"""

TEMPLATE_INFO = {
    # Artifacts
    "Pottery": {
        "file": "pottery.svg",
        "default_color": "#8B4513",
        "category": "artifacts"
    },
    "Stone Tool": {
        "file": "stone_tool.svg",
        "default_color": "#708090",
        "category": "artifacts"
    },
    "Bronze Artifact": {
        "file": "bronze.svg",
        "default_color": "#CD7F32",
        "category": "artifacts"
    },
    "Iron Artifact": {
        "file": "iron.svg",
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Ornament": {
        "file": "ornament.svg",
        "default_color": "#FFD700",
        "category": "artifacts"
    },
    "Coin": {
        "file": "coin.svg",
        "default_color": "#DAA520",
        "category": "artifacts"
    },
    "Bone Tool": {
        "file": "bone_tool.svg",
        "default_color": "#F5DEB3",
        "category": "artifacts"
    },
    "Weapon": {
        "file": "weapon.svg",
        "default_color": "#696969",
        "category": "artifacts"
    },
    "Arrowhead": {
        "file": "arrowhead.svg",
        "default_color": "#5F6A72",
        "category": "artifacts"
    },
    "Blade": {
        "file": "blade.svg",
        "default_color": "#50565D",
        "category": "artifacts"
    },
    "Scraper": {
        "file": "scraper.svg",
        "default_color": "#7A828B",
        "category": "artifacts"
    },
    "Needle / Pin": {
        "file": "needle_pin.svg",
        "default_color": "#8A7F73",
        "category": "artifacts"
    },
    "Bead": {
        "file": "bead.svg",
        "default_color": "#C68E3A",
        "category": "artifacts"
    },
    "Bracelet / Ring": {
        "file": "bracelet_ring.svg",
        "default_color": "#C9A227",
        "category": "artifacts"
    },
    "Seal / Stamp": {
        "file": "seal_stamp.svg",
        "default_color": "#8B5A2B",
        "category": "artifacts"
    },
    "Spindle Whorl": {
        "file": "spindle_whorl.svg",
        "default_color": "#7C5C46",
        "category": "artifacts"
    },
    "Chisel": {
        "file": "chisel.svg",
        "default_color": "#5B6168",
        "category": "artifacts"
    },
    "Bronze Dagger (Liaoning-style)": {
        "file": "bronze_dagger_liaoning.svg",
        "default_color": "#B66A62",
        "category": "artifacts"
    },
    "Bronze Dagger (Ordos-style)": {
        "file": "bronze_dagger_ordos.svg",
        "default_color": "#5E79B4",
        "category": "artifacts"
    },
    "Bronze Dagger (Antenna-style)": {
        "file": "bronze_dagger_antenna.svg",
        "default_color": "#4EA7A6",
        "category": "artifacts"
    },
    "Bronze Dagger (Slender)": {
        "file": "bronze_dagger_slender.svg",
        "default_color": "#B39A58",
        "category": "artifacts"
    },
    "Bronze Dagger (Tao type)": {
        "file": "bronze_dagger_tao.svg",
        "default_color": "#58A05A",
        "category": "artifacts"
    },
    "Bronze Dagger (Medium-fine)": {
        "file": "bronze_dagger_medium_fine.svg",
        "default_color": "#5E79B4",
        "category": "artifacts"
    },
    "Bronze Dagger (Flat bladed)": {
        "file": "bronze_dagger_flat_bladed.svg",
        "default_color": "#A06AC2",
        "category": "artifacts"
    },
    "Bronze Dagger (Type IA)": {
        "file": "bronze_dagger_type_ia.svg",
        "default_color": "#A3645C",
        "category": "artifacts"
    },
    "Bronze Dagger (Type IB)": {
        "file": "bronze_dagger_type_ib.svg",
        "default_color": "#8F7E5B",
        "category": "artifacts"
    },
    "Bronze Dagger (Other)": {
        "file": "bronze_dagger_other.svg",
        "default_color": "#8E8E8E",
        "category": "artifacts"
    },
    "Bronze Sword": {
        "file": "bronze_sword.svg",
        "default_color": "#53A9A8",
        "category": "artifacts"
    },
    "Bronze Dagger-axe": {
        "file": "bronze_dagger_axe.svg",
        "default_color": "#C66362",
        "category": "artifacts"
    },
    "Bronze Spear": {
        "file": "bronze_spear.svg",
        "default_color": "#B09657",
        "category": "artifacts"
    },
    "Pottery Rim Sherd (Section)": {
        "file": "pottery_rim_sherd_section.svg",
        "default_color": "#8B5A3C",
        "category": "artifacts"
    },
    "Pottery Base Sherd (Section)": {
        "file": "pottery_base_sherd_section.svg",
        "default_color": "#8B5A3C",
        "category": "artifacts"
    },
    "Pottery Body Sherd (Section)": {
        "file": "pottery_body_sherd_section.svg",
        "default_color": "#8B5A3C",
        "category": "artifacts"
    },
    "Projectile Point (Leaf-shaped)": {
        "file": "projectile_point_leaf.svg",
        "default_color": "#6E8FA3",
        "category": "artifacts"
    },
    "Projectile Point (Side-notched)": {
        "file": "projectile_point_side_notched.svg",
        "default_color": "#5B7FA2",
        "category": "artifacts"
    },
    "Projectile Point (Corner-notched)": {
        "file": "projectile_point_corner_notched.svg",
        "default_color": "#5B76A2",
        "category": "artifacts"
    },
    "Projectile Point (Stemmed)": {
        "file": "projectile_point_stemmed.svg",
        "default_color": "#667F95",
        "category": "artifacts"
    },
    "Projectile Point (Triangular)": {
        "file": "projectile_point_triangular.svg",
        "default_color": "#7F8FA1",
        "category": "artifacts"
    },

    # Structures
    "Fortress / Castle": {
        "file": "fortress.svg",
        "default_color": "#8B7355",
        "category": "structures"
    },
    "Dwelling / House": {
        "file": "dwelling.svg",
        "default_color": "#A0522D",
        "category": "structures"
    },
    "Tomb": {
        "file": "tomb.svg",
        "default_color": "#556B2F",
        "category": "structures"
    },
    "Keyhole Tomb (Normal)": {
        "file": "keyhole_tomb_normal.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Keyhole Tomb (With Moat)": {
        "file": "keyhole_tomb_with_moat.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Keyhole Tomb (Stepped)": {
        "file": "keyhole_tomb_stepped.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Keyhole Tomb (With Fukiishi)": {
        "file": "keyhole_tomb_with_fukiishi.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Keyhole Tomb (Tsumishizuka)": {
        "file": "keyhole_tomb_tsumishizuka.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Keyhole Tomb (Makinokuchi)": {
        "file": "keyhole_tomb_makinokuchi.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Normal)": {
        "file": "kofun_normal.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (With Shugo)": {
        "file": "kofun_with_shugo.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (With Fukiishi)": {
        "file": "kofun_with_fukiishi.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Tsumiishizuka)": {
        "file": "kofun_tsumiishizuka.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Enpun)": {
        "file": "kofun_enpun.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Zenpokouen)": {
        "file": "kofun_zenpokouen.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Makimuku-en)": {
        "file": "kofun_makimuku_en.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Hotategai)": {
        "file": "kofun_hotategai.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Sohochuen)": {
        "file": "kofun_sohochuen.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Hofun)": {
        "file": "kofun_hofun.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Zenpokoho)": {
        "file": "kofun_zenpokoho.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Makimuku-ho)": {
        "file": "kofun_makimuku_ho.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Yosumi)": {
        "file": "kofun_yosumi.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Kofun (Daijobo)": {
        "file": "kofun_daijobo.svg",
        "default_color": "#A88A5F",
        "category": "structures"
    },
    "Temple / Shrine": {
        "file": "temple.svg",
        "default_color": "#B22222",
        "category": "structures"
    },
    "Kiln / Furnace": {
        "file": "kiln.svg",
        "default_color": "#D2691E",
        "category": "structures"
    },
    "Well": {
        "file": "well.svg",
        "default_color": "#4682B4",
        "category": "structures"
    },
    "Wall / Rampart": {
        "file": "wall.svg",
        "default_color": "#808080",
        "category": "structures"
    },
    "Pit": {
        "file": "pit.svg",
        "default_color": "#6B4226",
        "category": "structures"
    },
    "Gate": {
        "file": "gate.svg",
        "default_color": "#8C6E4B",
        "category": "structures"
    },
    "Road / Pavement": {
        "file": "road_pavement.svg",
        "default_color": "#8A8A8A",
        "category": "structures"
    },
    "Bridge": {
        "file": "bridge.svg",
        "default_color": "#7A6C5D",
        "category": "structures"
    },
    "Storage Pit": {
        "file": "storage_pit.svg",
        "default_color": "#6F4F37",
        "category": "structures"
    },
    "Posthole": {
        "file": "posthole.svg",
        "default_color": "#5A4A3A",
        "category": "structures"
    },
    "Workshop": {
        "file": "workshop.svg",
        "default_color": "#8C5A3C",
        "category": "structures"
    },
    "Tower": {
        "file": "tower.svg",
        "default_color": "#707070",
        "category": "structures"
    },

    # Remains
    "Human Remains": {
        "file": "skull.svg",
        "default_color": "#DEB887",
        "category": "remains"
    },
    "Burial": {
        "file": "burial.svg",
        "default_color": "#8B8378",
        "category": "remains"
    },
    "Skeleton": {
        "file": "skeleton.svg",
        "default_color": "#C4A484",
        "category": "remains"
    },
    "Cremation Burial": {
        "file": "cremation_burial.svg",
        "default_color": "#A89F91",
        "category": "remains"
    },
    "Animal Remains": {
        "file": "animal_remains.svg",
        "default_color": "#BFA88D",
        "category": "remains"
    },

    # Features
    "Hearth / Fire Pit": {
        "draw": ("_draw_hearth", "COLOR"),
        "file": "hearth.svg",
        "default_color": "#FF4500",
        "category": "features"
    },
    "Midden / Shell Mound": {
        "file": "midden.svg",
        "default_color": "#BDB76B",
        "category": "features"
    },
    "Ditch / Moat": {
        "file": "ditch.svg",
        "default_color": "#2E8B57",
        "category": "features"
    },
    "Stone Alignment": {
        "file": "stone_align.svg",
        "default_color": "#778899",
        "category": "features"
    },
    "Dolmen": {
        "file": "dolmen.svg",
        "default_color": "#A9A9A9",
        "category": "features"
    },
    "Rock Art": {
        "file": "rock_art.svg",
        "default_color": "#CD853F",
        "category": "features"
    },
    "Canal / Water Channel": {
        "file": "canal_water_channel.svg",
        "default_color": "#3B7EA1",
        "category": "features"
    },
    "Terrace": {
        "file": "terrace.svg",
        "default_color": "#8A7760",
        "category": "features"
    },
    "Ash Layer": {
        "file": "ash_layer.svg",
        "default_color": "#7D7D7D",
        "category": "features"
    },
    "Burnt Area": {
        "file": "burnt_area.svg",
        "default_color": "#6A4E42",
        "category": "features"
    },
    "Mound / Barrow": {
        "file": "mound_barrow.svg",
        "default_color": "#7A6A50",
        "category": "features"
    },
    "Standing Stone": {
        "file": "standing_stone.svg",
        "default_color": "#8A9096",
        "category": "features"
    },

    # Survey
    "Excavation Area": {
        "file": "excavation.svg",
        "default_color": "#FF8C00",
        "category": "survey"
    },
    "Survey Point": {
        "file": "survey_point.svg",
        "default_color": "#4169E1",
        "category": "survey"
    },
    "Find Spot": {
        "file": "find_spot.svg",
        "default_color": "#DC143C",
        "category": "survey"
    },
    "Trench": {
        "file": "trench.svg",
        "default_color": "#D97706",
        "category": "survey"
    },
    "Datum Point": {
        "file": "datum_point.svg",
        "default_color": "#1D4ED8",
        "category": "survey"
    },
    "Sample Location": {
        "file": "sample_location.svg",
        "default_color": "#BE123C",
        "category": "survey"
    },
    "Photo Point": {
        "file": "photo_point.svg",
        "default_color": "#7C3AED",
        "category": "survey"
    },
    "Grid Corner": {
        "file": "grid_corner.svg",
        "default_color": "#0F766E",
        "category": "survey"
    },
    "Test Pit": {
        "file": "test_pit.svg",
        "default_color": "#92400E",
        "category": "survey"
    },
    "North Arrow (Map Standard)": {
        "file": "north_arrow_map_standard.svg",
        "default_color": "#1F2937",
        "category": "survey"
    },
    "Scale Bar (Map Standard)": {
        "file": "scale_bar_map_standard.svg",
        "default_color": "#1F2937",
        "category": "survey"
    },
    "Harris Matrix Context": {
        "file": "harris_matrix_context.svg",
        "default_color": "#6B7280",
        "category": "survey"
    },
    "Stratigraphic Unit": {
        "file": "stratigraphic_unit.svg",
        "default_color": "#4B5563",
        "category": "survey"
    },
    # -- Korean tomb types (한국 무덤) --------------------------------
    "Dolmen (Table-type)": {
        "draw": ("_draw_korean_tomb", "table", "COLOR"),
        "default_color": "#708090",
        "category": "structures"
    },
    "Dolmen (Go-board-type)": {
        "draw": ("_draw_korean_tomb", "go_board", "COLOR"),
        "default_color": "#708090",
        "category": "structures"
    },
    "Dolmen (Capstone-type)": {
        "draw": ("_draw_korean_tomb", "capstone", "COLOR"),
        "default_color": "#708090",
        "category": "structures"
    },
    "Stone Cist Tomb": {
        "draw": ("_draw_korean_tomb", "stone_cist", "COLOR"),
        "default_color": "#7A8288",
        "category": "structures"
    },
    "Stone-lined Tomb": {
        "draw": ("_draw_korean_tomb", "stone_lined", "COLOR"),
        "default_color": "#7A8288",
        "category": "structures"
    },
    "Wooden Coffin Tomb": {
        "draw": ("_draw_korean_tomb", "wooden_coffin", "COLOR"),
        "default_color": "#8B6F47",
        "category": "structures"
    },
    "Wooden Chamber Tomb": {
        "draw": ("_draw_korean_tomb", "wooden_chamber", "COLOR"),
        "default_color": "#8B6F47",
        "category": "structures"
    },
    "Jar Coffin Tomb": {
        "draw": ("_draw_korean_tomb", "jar_coffin", "COLOR"),
        "default_color": "#A0522D",
        "category": "structures"
    },
    "Stone-mounded Wooden Chamber Tomb": {
        "draw": ("_draw_korean_tomb", "stone_mound_chamber", "COLOR"),
        "default_color": "#6E7B8B",
        "category": "structures"
    },
    "Corridor-style Stone Chamber Tomb": {
        "draw": ("_draw_korean_tomb", "corridor_chamber", "COLOR"),
        "default_color": "#778899",
        "category": "structures"
    },
    "Earthen Mounded Tomb": {
        "draw": ("_draw_korean_tomb", "earthen_mound", "COLOR"),
        "default_color": "#9C8B6E",
        "category": "structures"
    },
    "Ditch-encircled Tomb": {
        "draw": ("_draw_korean_tomb", "ditch_encircled", "COLOR"),
        "default_color": "#8B7355",
        "category": "structures"
    },
    "Earthen Pit Tomb": {
        "draw": ("_draw_korean_tomb", "pit_grave", "COLOR"),
        "default_color": "#8B7D6B",
        "category": "structures"
    },
    # -- Korean settlement, production and defence features -----------
    "Pit Dwelling (Round)": {
        "draw": ("_draw_korean_feature", "pit_house_round", "COLOR"),
        "default_color": "#A0764B",
        "category": "structures"
    },
    "Pit Dwelling (Square)": {
        "draw": ("_draw_korean_feature", "pit_house_square", "COLOR"),
        "default_color": "#A0764B",
        "category": "structures"
    },
    "Pit Dwelling (Protruding Entrance)": {
        "draw": ("_draw_korean_feature", "pit_house_convex", "COLOR"),
        "default_color": "#A0764B",
        "category": "structures"
    },
    "Pit Dwelling (Twin-room)": {
        "draw": ("_draw_korean_feature", "pit_house_twin", "COLOR"),
        "default_color": "#A0764B",
        "category": "structures"
    },
    "Raised-floor Building": {
        "draw": ("_draw_korean_feature", "raised_floor", "COLOR"),
        "default_color": "#8B7355",
        "category": "structures"
    },
    "Cooking Stove (Kamado)": {
        "draw": ("_draw_korean_feature", "kamado", "COLOR"),
        "default_color": "#B4531F",
        "category": "features"
    },
    "Ondol Heating Flue": {
        "draw": ("_draw_korean_feature", "ondol", "COLOR"),
        "default_color": "#B4531F",
        "category": "features"
    },
    "Pottery Kiln": {
        "draw": ("_draw_korean_feature", "pottery_kiln", "COLOR"),
        "default_color": "#C1440E",
        "category": "structures"
    },
    "Roof Tile Kiln": {
        "draw": ("_draw_korean_feature", "tile_kiln", "COLOR"),
        "default_color": "#C1440E",
        "category": "structures"
    },
    "Iron Smelting Feature": {
        "draw": ("_draw_korean_feature", "iron_smelting", "COLOR"),
        "default_color": "#5A5A5A",
        "category": "structures"
    },
    "Charcoal Kiln": {
        "draw": ("_draw_korean_feature", "charcoal_kiln", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "structures"
    },
    "Paddy Field": {
        "draw": ("_draw_korean_feature", "paddy_field", "COLOR"),
        "default_color": "#6B8E23",
        "category": "features"
    },
    "Dry Field": {
        "draw": ("_draw_korean_feature", "dry_field", "COLOR"),
        "default_color": "#8B7B3A",
        "category": "features"
    },
    "Earthen Rampart Fortress": {
        "draw": ("_draw_korean_feature", "earthen_rampart", "COLOR"),
        "default_color": "#7B6A4F",
        "category": "structures"
    },
    "Stone Rampart Fortress": {
        "draw": ("_draw_korean_feature", "stone_rampart", "COLOR"),
        "default_color": "#778899",
        "category": "structures"
    },
    "Mountain Fortress": {
        "draw": ("_draw_korean_feature", "mountain_fortress", "COLOR"),
        "default_color": "#6B705C",
        "category": "structures"
    },
    "Palisade": {
        "draw": ("_draw_korean_feature", "palisade", "COLOR"),
        "default_color": "#8B6F47",
        "category": "structures"
    },
    "Encircling Ditch": {
        "draw": ("_draw_korean_feature", "encircling_ditch", "COLOR"),
        "default_color": "#708090",
        "category": "features"
    },
    "Beacon Station": {
        "draw": ("_draw_korean_feature", "beacon", "COLOR"),
        "default_color": "#C1440E",
        "category": "structures"
    },
    "Water Collection Basin": {
        "draw": ("_draw_korean_feature", "water_basin", "COLOR"),
        "default_color": "#4682B4",
        "category": "structures"
    },
    # -- Korean pottery and ceramics (토기·도자기) ---------------------
    "Comb-pattern Pottery": {
        "draw": ("_draw_korean_pottery", "comb_pattern", "COLOR"),
        "default_color": "#A9825E",
        "category": "artifacts"
    },
    "Plain Coarse Pottery": {
        "draw": ("_draw_korean_pottery", "plain_coarse", "COLOR"),
        "default_color": "#B08968",
        "category": "artifacts"
    },
    "Red Burnished Pottery": {
        "draw": ("_draw_korean_pottery", "red_burnished", "COLOR"),
        "default_color": "#A5402B",
        "category": "artifacts"
    },
    "Black Burnished Long-necked Jar": {
        "draw": ("_draw_korean_pottery", "black_burnished", "COLOR"),
        "default_color": "#3B3B3B",
        "category": "artifacts"
    },
    "Soft Grey Pottery (Wajil)": {
        "draw": ("_draw_korean_pottery", "wajil", "COLOR"),
        "default_color": "#8B8B83",
        "category": "artifacts"
    },
    "Hard Grey Stoneware (Gyeongjil)": {
        "draw": ("_draw_korean_pottery", "gyeongjil", "COLOR"),
        "default_color": "#6E6E6E",
        "category": "artifacts"
    },
    "Mounted Dish (Gobae)": {
        "draw": ("_draw_korean_pottery", "gobae", "COLOR"),
        "default_color": "#9C7A56",
        "category": "artifacts"
    },
    "Storage Jar (Ho)": {
        "draw": ("_draw_korean_pottery", "storage_jar", "COLOR"),
        "default_color": "#8B5A2B",
        "category": "artifacts"
    },
    "Steamer (Siru)": {
        "draw": ("_draw_korean_pottery", "siru", "COLOR"),
        "default_color": "#A0764B",
        "category": "artifacts"
    },
    "Celadon": {
        "draw": ("_draw_korean_pottery", "celadon", "COLOR"),
        "default_color": "#4E8C7E",
        "category": "artifacts"
    },
    "Buncheong Ware": {
        "draw": ("_draw_korean_pottery", "buncheong", "COLOR"),
        "default_color": "#9BA69B",
        "category": "artifacts"
    },
    "White Porcelain": {
        "draw": ("_draw_korean_pottery", "white_porcelain", "COLOR"),
        "default_color": "#C9C6BC",
        "category": "artifacts"
    },
    "Onggi Jar": {
        "draw": ("_draw_korean_pottery", "onggi", "COLOR"),
        "default_color": "#5B3A22",
        "category": "artifacts"
    },
    # -- Korean stone, bronze and iron tools (석기·청동기·철기) ---------
    "Handaxe": {
        "draw": ("_draw_korean_tool", "handaxe", "COLOR"),
        "default_color": "#708090",
        "category": "artifacts"
    },
    "Chopper": {
        "draw": ("_draw_korean_tool", "chopper", "COLOR"),
        "default_color": "#7A8288",
        "category": "artifacts"
    },
    "Tanged Point": {
        "draw": ("_draw_korean_tool", "tanged_point", "COLOR"),
        "default_color": "#708090",
        "category": "artifacts"
    },
    "Microblade Core": {
        "draw": ("_draw_korean_tool", "microblade_core", "COLOR"),
        "default_color": "#7A8288",
        "category": "artifacts"
    },
    "Polished Stone Dagger": {
        "draw": ("_draw_korean_tool", "polished_dagger", "COLOR"),
        "default_color": "#6E7B8B",
        "category": "artifacts"
    },
    "Semi-lunar Stone Knife": {
        "draw": ("_draw_korean_tool", "semilunar_knife", "COLOR"),
        "default_color": "#708090",
        "category": "artifacts"
    },
    "Stone Hoe": {
        "draw": ("_draw_korean_tool", "stone_hoe", "COLOR"),
        "default_color": "#7A8288",
        "category": "artifacts"
    },
    "Grinding Slab and Muller": {
        "draw": ("_draw_korean_tool", "grinding_slab", "COLOR"),
        "default_color": "#8B8378",
        "category": "artifacts"
    },
    "Stone Arrowhead": {
        "draw": ("_draw_korean_tool", "stone_arrowhead", "COLOR"),
        "default_color": "#708090",
        "category": "artifacts"
    },
    "Net Sinker": {
        "draw": ("_draw_korean_tool", "net_sinker", "COLOR"),
        "default_color": "#7A8288",
        "category": "artifacts"
    },
    "Coarse-lined Bronze Mirror": {
        "draw": ("_draw_korean_tool", "coarse_mirror", "COLOR"),
        "default_color": "#CD7F32",
        "category": "artifacts"
    },
    "Fine-lined Bronze Mirror": {
        "draw": ("_draw_korean_tool", "fine_mirror", "COLOR"),
        "default_color": "#CD7F32",
        "category": "artifacts"
    },
    "Bronze Rattle": {
        "draw": ("_draw_korean_tool", "bronze_rattle", "COLOR"),
        "default_color": "#CD7F32",
        "category": "artifacts"
    },
    "Bronze Bell": {
        "draw": ("_draw_korean_tool", "bronze_bell", "COLOR"),
        "default_color": "#CD7F32",
        "category": "artifacts"
    },
    "Iron Sword": {
        "draw": ("_draw_korean_tool", "iron_sword", "COLOR"),
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Iron Spearhead": {
        "draw": ("_draw_korean_tool", "iron_spearhead", "COLOR"),
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Iron Arrowhead": {
        "draw": ("_draw_korean_tool", "iron_arrowhead", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "artifacts"
    },
    "Iron Axe": {
        "draw": ("_draw_korean_tool", "iron_axe", "COLOR"),
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Iron Ard": {
        "draw": ("_draw_korean_tool", "iron_ard", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "artifacts"
    },
    "Iron Sickle": {
        "draw": ("_draw_korean_tool", "iron_sickle", "COLOR"),
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Plate Armour": {
        "draw": ("_draw_korean_tool", "plate_armour", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "artifacts"
    },
    "Lamellar Armour": {
        "draw": ("_draw_korean_tool", "lamellar_armour", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "artifacts"
    },
    "Horse Bit": {
        "draw": ("_draw_korean_tool", "horse_bit", "COLOR"),
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Stirrup": {
        "draw": ("_draw_korean_tool", "stirrup", "COLOR"),
        "default_color": "#434343",
        "category": "artifacts"
    },
    "Iron Ingot": {
        "draw": ("_draw_korean_tool", "iron_ingot", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "artifacts"
    },
    # -- Korean ornaments, tiles and other finds (장신구·기와·기타) -----
    "Comma-shaped Jade (Gogok)": {
        "draw": ("_draw_korean_ornament", "gogok", "COLOR"),
        "default_color": "#4E8C7E",
        "category": "artifacts"
    },
    "Tubular Jade Bead (Gwanok)": {
        "draw": ("_draw_korean_ornament", "gwanok", "COLOR"),
        "default_color": "#6E8B74",
        "category": "artifacts"
    },
    "Glass Bead": {
        "draw": ("_draw_korean_ornament", "glass_bead", "COLOR"),
        "default_color": "#3C6E9C",
        "category": "artifacts"
    },
    "Gold Earring": {
        "draw": ("_draw_korean_ornament", "gold_earring", "COLOR"),
        "default_color": "#C9A227",
        "category": "artifacts"
    },
    "Gold Crown": {
        "draw": ("_draw_korean_ornament", "gold_crown", "COLOR"),
        "default_color": "#C9A227",
        "category": "artifacts"
    },
    "Belt Fitting Set": {
        "draw": ("_draw_korean_ornament", "belt_fitting", "COLOR"),
        "default_color": "#B8860B",
        "category": "artifacts"
    },
    "Wooden Document Slip (Mokgan)": {
        "draw": ("_draw_korean_ornament", "mokgan", "COLOR"),
        "default_color": "#A98B62",
        "category": "artifacts"
    },
    "Round Roof-end Tile": {
        "draw": ("_draw_korean_ornament", "round_roof_tile", "COLOR"),
        "default_color": "#8B7355",
        "category": "artifacts"
    },
    "Eaves Roof Tile": {
        "draw": ("_draw_korean_ornament", "eaves_roof_tile", "COLOR"),
        "default_color": "#8B7355",
        "category": "artifacts"
    },
    "Floor Brick": {
        "draw": ("_draw_korean_ornament", "floor_brick", "COLOR"),
        "default_color": "#9C7A56",
        "category": "artifacts"
    },
    "Inkstone": {
        "draw": ("_draw_korean_ornament", "inkstone", "COLOR"),
        "default_color": "#4A4A4A",
        "category": "artifacts"
    },
    "Clay Figurine": {
        "draw": ("_draw_korean_ornament", "clay_figurine", "COLOR"),
        "default_color": "#A0764B",
        "category": "artifacts"
    },
    "Ridge-end Roof Ornament (Chimi)": {
        "draw": ("_draw_korean_ornament", "chimi", "COLOR"),
        "default_color": "#7A6A56",
        "category": "artifacts"
    },
    "Building Foundation Stone": {
        "draw": ("_draw_korean_ornament", "foundation_stone", "COLOR"),
        "default_color": "#808080",
        "category": "structures"
    },
}


# Backward compatibility for older naming variants
LEGACY_TEMPLATE_ALIASES = {
    "Stone Tools": "Stone Tool",
    "Bronze Artifacts": "Bronze Artifact",
    "Iron Artifacts": "Iron Artifact",
    "Ornaments": "Ornament",
    "Coins": "Coin",
    "Bone/Antler Tools": "Bone Tool",
    "Weapons": "Weapon",
    "Fortress/Castle": "Fortress / Castle",
    "Dwelling/House": "Dwelling / House",
    "Tomb/Burial": "Tomb",
    "Temple/Shrine": "Temple / Shrine",
    "Kiln/Furnace": "Kiln / Furnace",
    "Wall/Rampart": "Wall / Rampart",
    "Hearth/Fire Pit": "Hearth / Fire Pit",
    "Midden/Shell Mound": "Midden / Shell Mound",
    "Ditch/Moat": "Ditch / Moat",
    "Liaoning-style bronze dagger": "Bronze Dagger (Liaoning-style)",
    "Ordos-style bronze dagger": "Bronze Dagger (Ordos-style)",
    "Antenna-style bronze dagger": "Bronze Dagger (Antenna-style)",
    "Slender bronze dagger": "Bronze Dagger (Slender)",
    "Tao Shi Jian sword": "Bronze Dagger (Tao type)",
    "Medium-fine bronze sword": "Bronze Dagger (Medium-fine)",
    "Flat bladed bronze sword": "Bronze Dagger (Flat bladed)",
    "Type IA bronze dagger": "Bronze Dagger (Type IA)",
    "Type IB bronze dagger": "Bronze Dagger (Type IB)",
    "Other bronze sword": "Bronze Dagger (Other)",
    "Leppy Hills point": "Projectile Point (Leaf-shaped)",
    "Pequop side-notched point": "Projectile Point (Side-notched)",
    "Dead Cedar point": "Projectile Point (Corner-notched)",
    "Elko-eared point": "Projectile Point (Stemmed)",
    "Normal": "Kofun (Normal)",
    "with Shugo": "Keyhole Tomb (With Moat)",
    "with moat": "Keyhole Tomb (With Moat)",
    "with Fukiishi": "Keyhole Tomb (With Fukiishi)",
    "Tsumishizuka": "Keyhole Tomb (Tsumishizuka)",
    "Tsumiishizuka": "Kofun (Tsumiishizuka)",
    "Makinokuchi": "Keyhole Tomb (Makinokuchi)",
    "Enpun": "Kofun (Enpun)",
    "Zenpokouen": "Kofun (Zenpokouen)",
    "Zenpokoen": "Kofun (Zenpokouen)",
    "Makimuku-en": "Kofun (Makimuku-en)",
    "Hotategai": "Kofun (Hotategai)",
    "Sohochuen": "Kofun (Sohochuen)",
    "Hofun": "Kofun (Hofun)",
    "Zenpokoho": "Kofun (Zenpokoho)",
    "Makimuku-ho": "Kofun (Makimuku-ho)",
    "Yosumi": "Kofun (Yosumi)",
    "Daijobo": "Kofun (Daijobo)",
    "Rim Sherd": "Pottery Rim Sherd (Section)",
    "Base Sherd": "Pottery Base Sherd (Section)",
    "Body Sherd": "Pottery Body Sherd (Section)",
    "North Arrow": "North Arrow (Map Standard)",
    "Scale Bar": "Scale Bar (Map Standard)",
    "Harris Context": "Harris Matrix Context",
    "Harris Matrix": "Harris Matrix Context",
    "Stratigraphic Unit Symbol": "Stratigraphic Unit",
}
