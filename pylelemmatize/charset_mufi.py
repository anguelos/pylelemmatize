# code automatically generated look at experiments/mufi_extraction/mufi_extractor.ipynb

from typing import Tuple, Dict, List, Optional


# PUBLIC = PUBLIC
# PRIVATE = PUBLIC U PRIVATE
# BMP = BMP
# NONBMP = BMP U NONBMP

def get_encoding_dicts() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    __public_bmp_mufi_str = ' !"#$%&\'()*+,-./0123' + \
    '456789:;<=>?@ABCDEFG' + \
    'HIJKLMNOPQRSTUVWXYZ[' + \
    '\\]^_`abcdefghijklmno' + \
    'pqrstuvwxyz{|}~\x7f\xa0¡¢£' + \
    '¤¥¦§¨©ª«¬\xad®¯°±²³´µ¶·' + \
    '¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊË' + \
    'ÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß' + \
    'àáâãäåæçèéêëìíîïðñòó' + \
    'ôõö÷øùúûüýþÿĀāĂăĄąĆć' + \
    'ĊċĐđĒēĔĕĖėĘęĠġħĪīĬĭĮ' + \
    'įİıĲĳĹĺŁłŃńŊŋŌōŎŏŐőŒ' + \
    'œŔŕŚśŪūŬŭŮůŰűŲųŴŵŶŷŸ' + \
    'ŻżſƀƕƙƚƜƞƦƵƶƷƿǑǒǓǔǕǖ' + \
    'ǢǣǤǥǪǫǬǭǴǵǶǷǼǽǾǿȜȝȦȧ' + \
    'ȪȫȮȯȲȳȷɈɉɔɖəɟɡɢɦɨɪɯɲ' + \
    'ɴɶɼʀʉʏʒʙʜʟʻʼˈ˘˙˚˛˜˝ˣ' + \
    '̧̨̣̲̳̀́̂̃̄̅̆̇̈̉̊̋̍̎̕' + \
    '̶͙̾̿͂͛ͣͤͥͦͧͨͩͪͫͬͭͮͯ͜' + \
    'Θθස჻ᚠᛘᴀᴁᴄᴅᴆᴇᴊᴋᴍᴏᴘᴛᴜᴠ' + \
    'ᴡᴢᴵᵫᵹ᷐᷎᷏᷑᷒ᷓᷔᷕᷖᷗᷘᷙᷚᷛ᷍' + \
    'ᷜᷝᷞᷟᷠᷡᷢᷣᷤᷥᷦḂḃḄḅḊḋḌḍḖ' + \
    'ḗḞḟḢḣḤḥḰḱḲḳḶḷḾḿṀṁṂṃṄ' + \
    'ṅṆṇṒṓṔṕṖṗṘṙṚṛṠṡṢṣṪṫṬ' + \
    'ṭṾṿẀẁẂẃẄẅẆẇẈẉẎẏẒẓẘẙẜ' + \
    'ẝẞẟẠạẢảẮắẸẹỈỉỊịỌọỎỏỤ' + \
    'ụỦủỲỳỴỵỶỷỺỻỼỽỾỿ\u2000\u2001\u2002\u2003\u2004' + \
    '\u2005\u2006\u2007\u2008\u2009\u200a\u200b‐‑‒–—―‖‘’‚‛“”' + \
    '„‟†‡•‣․‥…‧\u202f‰′″‹›※⁂⁄⁅' + \
    '⁆⁊⁋⁖⁘⁜⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆' + \
    '₇₈₉₰₻℈℔℞℟℣℥ℲⅎↀↁↂↃↄ←↑' + \
    '→↓−∞∧∴∵∷∻≈≠⋗⏑⏒⏓⏔▪▫▹◃' + \
    '◌✝❦❧⟦⟧⟨⟩⫽Ⱶⱶⱽ⸀⸌⸍⸗⸙⸜⸝⸠' + \
    '⸡⸢⸣⸤⸥⸦⸧⸨⸩⸪⸫⸬⸭⸮⹀ꜰꜱꜲꜳꜴ' + \
    'ꜵꜶꜷꜸꜹꜺꜻꜼꜽꜾꜿꝀꝁꝂꝃꝄꝅꝆꝇꝈ' + \
    'ꝉꝊꝋꝌꝍꝎꝏꝐꝑꝒꝓꝔꝕꝖꝗꝘꝙꝚꝛꝜ' + \
    'ꝝꝞꝟꝠꝡꝢꝣꝤꝥꝦꝧꝨꝩꝪꝫꝬꝭꝮꝯꝰ' + \
    'ꝱꝲꝳꝴꝵꝶꝷꝸꝹꝺꝻꝼꝽꝾꝿꞀꞁꞂꞃꞄ' + \
    'ꞅꞆꞇꟻꟼꟽꟾꟿꭗ곎ﬀﬁﬂﬃﬄﬅﬆ'

    __public_nonbmp_mufi_str = '𐆐𐆑𐆒𐆓𐆔𐆕𐆖𐆗𐆘𐆙𐆚'

    __private_bmp_mufi_str = '\ue004\ue00a\ue010\ue025\ue02c\ue033\ue036\ue03a\ue03d\ue03f\ue040\ue041\ue042\ue043\ue044\ue066\ue076\ue077\ue08f\ue099' + \
    '\ue0b7\ue0bc\ue0c8\ue0d1\ue0e1\ue0e8\ue0e9\ue0ea\ue0eb\ue0ec\ue0ee\ue0f0\ue101\ue116\ue12a\ue135\ue137\ue143\ue150\ue151' + \
    '\ue152\ue153\ue154\ue15c\ue162\ue163\ue168\ue19e\ue1b8\ue1d2\ue1dc\ue208\ue20c\ue21b\ue22d\ue244\ue246\ue24f\ue252\ue253' + \
    '\ue255\ue257\ue259\ue25d\ue260\ue262\ue268\ue26d\ue282\ue288\ue2e2\ue2ee\ue309\ue30b\ue315\ue317\ue324\ue32b\ue32d\ue331' + \
    '\ue337\ue33a\ue33b\ue342\ue34b\ue34c\ue34d\ue34e\ue350\ue353\ue357\ue373\ue375\ue376\ue37c\ue384\ue385\ue39f\ue3d3\ue3d4' + \
    '\ue3e5\ue3e6\ue3e7\ue404\ue40a\ue410\ue41a\ue41d\ue41f\ue425\ue42c\ue42d\ue42e\ue433\ue436\ue43a\ue43d\ue43f\ue440\ue441' + \
    '\ue442\ue443\ue444\ue44d\ue466\ue476\ue477\ue48f\ue491\ue498\ue499\ue49f\ue4b7\ue4bc\ue4c8\ue4cd\ue4cf\ue4d1\ue4e1\ue4e2' + \
    '\ue4e3\ue4e8\ue4e9\ue4ea\ue4eb\ue4ec\ue4ee\ue4f0\ue501\ue516\ue517\ue52a\ue535\ue537\ue543\ue548\ue54a\ue54b\ue550\ue551' + \
    '\ue552\ue553\ue554\ue562\ue563\ue568\ue58c\ue596\ue59e\ue5a4\ue5b1\ue5b8\ue5c5\ue5d2\ue5d7\ue5dc\ue5ee\ue608\ue60c\ue60e' + \
    '\ue61b\ue62c\ue62d\ue637\ue643\ue644\ue645\ue646\ue647\ue64f\ue652\ue653\ue655\ue657\ue659\ue65d\ue660\ue662\ue665\ue668' + \
    '\ue66d\ue681\ue682\ue688\ue68b\ue6a3\ue6e2\ue6ee\ue709\ue70b\ue715\ue717\ue724\ue727\ue72b\ue72c\ue72d\ue731\ue734\ue735' + \
    '\ue737\ue73a\ue73b\ue742\ue743\ue74b\ue74c\ue74d\ue74e\ue74f\ue750\ue753\ue754\ue757\ue773\ue775\ue776\ue77b\ue77c\ue781' + \
    '\ue784\ue785\ue79e\ue79f\ue7a2\ue7b2\ue7c1\ue7c2\ue7c3\ue7c7\ue7c8\ue7cc\ue7d3\ue7d4\ue7e4\ue7e5\ue7e6\ue7e7\ue8a1\ue8a2' + \
    '\ue8a3\ue8b3\ue8b4\ue8b7\ue8b8\ue8ba\ue8bb\ue8bc\ue8bd\ue8be\ue8bf\ue8c1\ue8c2\ue8c3\ue8c5\ue8c6\ue8c7\ue8c8\ue8c9\ue8ce' + \
    '\ue8d1\ue8d3\ue8d5\ue8d7\ue8dd\ue8de\ue8df\ue8e0\ue8e1\ue8e2\ue8e3\ue8e4\ue8e5\ue8e6\ue8e7\ue8e8\ue8e9\ue8ea\ue8eb\ue8ec' + \
    '\ue8ed\ue8f0\ue8f1\ue8f2\ue8f3\uead0\uead1\uead2\ueada\ueaf0\ueaf2\ueaf3\ueba0\ueba1\ueba2\ueba3\ueba4\ueba5\ueba6\ueba7' + \
    '\ueba8\ueba9\uebaa\uebab\uebac\uebad\uebae\uebaf\uebb0\uebb1\uebb2\uebb3\uebb4\uebb5\uebb6\uebb7\uebb8\uebb9\uebba\uebbb' + \
    '\uebbd\uebbe\uebbf\uebc0\uebc1\uebc2\uebc3\uebc4\uebc5\uebc6\uebc7\uebc8\uebc9\uebca\uebcb\uebcd\uebce\uebcf\uebd0\uebd1' + \
    '\uebd2\uebd3\uebd4\uebd5\uebd6\uebd7\uebda\uebdb\uebdc\uebdd\uebde\uebdf\uebe0\uebe1\uebe2\uebe3\uebe4\uebe5\uebe6\uebe7' + \
    '\uebe8\uebe9\uebea\uebeb\uebec\uebed\uebee\uebef\uebf0\uebf1\uebf2\uebf3\uebf4\uebf5\uebf6\uebf7\uebf8\uebf9\uebfa\uebfb' + \
    '\uebfc\uebfd\uebfe\uebff\ueec2\ueec3\ueec4\ueec5\ueec6\ueec7\ueec8\ueec9\ueeca\ueecb\ueecc\ueecd\ueece\ueecf\ueed0\ueed1' + \
    '\ueed2\ueed3\ueed4\ueed5\ueed6\ueed7\ueed8\ueed9\ueeda\ueedb\ueedc\ueedd\ueede\ueedf\ueee0\ueee1\ueee2\ueee3\ueee4\ueee5' + \
    '\ueee6\ueee7\ueee8\ueee9\ueeea\ueeeb\ueeec\ueeed\ueeee\ueeef\ueef0\ueef1\ueef2\ueef3\ueef4\ueef5\ueef6\ueef7\ueef8\ueef9' + \
    '\ueefa\ueefb\ueefc\ueefd\ueefe\ueeff\uef0c\uef11\uef15\uef20\uef21\uef22\uef23\uef24\uef25\uef26\uef27\uef28\uef29\uef2a' + \
    '\uef2b\uef2c\uef2d\uefa0\uefa1\uefa2\uefa3\uefa4\uefa5\uefa6\uefa7\uefa8\uefa9\uefaa\uefab\uefac\uefad\uefae\uefd8\uefd9' + \
    '\uefdb\uefdc\uefdd\uefde\uefdf\uefe0\uefe1\uefe2\uefe3\uefe4\uefe5\uefe6\uefe7\uefe8\uefe9\uefea\uefeb\uefec\uefed\uefee' + \
    '\uefef\ueff0\ueff1\ueff2\ueff3\ueff4\ueff5\ueff6\ueff7\ueff8\ueff9\ueffa\ueffb\ueffc\ueffd\ueffe\uefff\uf00a\uf00b\uf00c' + \
    '\uf00d\uf012\uf013\uf016\uf017\uf01c\uf025\uf02a\uf02b\uf02f\uf030\uf031\uf032\uf033\uf036\uf038\uf03a\uf03b\uf03c\uf03d' + \
    '\uf03e\uf03f\uf040\uf106\uf10a\uf10e\uf110\uf11a\uf126\uf127\uf128\uf130\uf135\uf136\uf13a\uf13e\uf13f\uf142\uf149\uf153' + \
    '\uf158\uf159\uf160\uf161\uf193\uf194\uf195\uf196\uf198\uf199\uf19a\uf19b\uf1a5\uf1a6\uf1a7\uf1ac\uf1bb\uf1bc\uf1bd\uf1bf' + \
    '\uf1c0\uf1c1\uf1c2\uf1c5\uf1c7\uf1c8\uf1ca\uf1cc\uf1d2\uf1da\uf1e0\uf1e1\uf1e2\uf1e3\uf1e4\uf1e5\uf1e6\uf1e7\uf1e8\uf1ea' + \
    '\uf1ec\uf1f0\uf1f1\uf1f2\uf1f4\uf1f5\uf1f7\uf1f8\uf1f9\uf1fa\uf1fb\uf1fc\uf200\uf201\uf202\uf203\uf204\uf205\uf206\uf207' + \
    '\uf208\uf209\uf214\uf215\uf217\uf218\uf219\uf21a\uf21b\uf21c\uf21d\uf21e\uf21f\uf220\uf221\uf222\uf223\uf224\uf225\uf226' + \
    '\uf228\uf229\uf22a\uf22b\uf22c\uf233\uf23a\uf23c\uf23d\uf23e\uf23f\uf2e0\uf2e2\uf2e3\uf2e4\uf2e6\uf2e7\uf2e8\uf2e9\uf2ea' + \
    '\uf2eb\uf2ec\uf2ed\uf2ee\uf2ef\uf2f0\uf2f1\uf2f2\uf2f3\uf2f4\uf2f5\uf2f6\uf2f7\uf2f8\uf2f9\uf2fa\uf2fb\uf2fd\uf2fe\uf2ff' + \
    '\uf4f9\uf4fa\uf4fb\uf4fc\uf4fd\uf4fe\uf4ff\uf704\uf7b2\uf7b3\uf7b4\uf7b5\uf7b6'

    __private_nonbmp_mufi_str = ''

    all_bmp_mufi = {'bmp_mufi': __public_bmp_mufi_str, 'bmp_pua_mufi': __public_bmp_mufi_str +  __private_bmp_mufi_str}
    all_nonbmp_mufi = {k:v for k, v in all_bmp_mufi.items()}
    all_nonbmp_mufi.update({'nonbmp_mufi': __public_nonbmp_mufi_str, 'nonbmp_pua_mufi': __public_nonbmp_mufi_str +  __private_nonbmp_mufi_str})
    return all_bmp_mufi, all_nonbmp_mufi

#all_bmp_mufi, all_nonbmp_mufi = get_encoding_dicts()

all_mufibmp_alphabets = ["bmp_mufi", "bmp_pua_mufi"]
all_mufi_alphabets = ["nonbmp_mufi", "bmp_mufi", "bmp_pua_mufi", "nonbmp_pua_mufi"]