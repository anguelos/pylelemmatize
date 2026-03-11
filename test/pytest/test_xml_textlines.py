import pytest
from pylelemmatize import XMLTextlines
from lxml import etree


def test_read_alto_texts():
    alto_extractor = XMLTextlines.from_alto(alto_xml_str)
    assert all([txt == alto_str_textlines[i] for i, txt in enumerate(alto_extractor)])
    str_et = etree.fromstring(alto_xml_str.encode("utf-8"))
    normalised_xml_str = etree.tostring(str_et, method="c14n").decode("utf-8")
    assert normalised_xml_str == alto_extractor.get_xml_str()

def test_read_page_xml_texts():
    page_extractor = XMLTextlines.from_pagexml(page_xml_str)
    assert all([txt == page_xml_str_textlines[i] for i, txt in enumerate(page_extractor)])
    str_et = etree.fromstring(page_xml_str.encode("utf-8"))
    normalised_xml_str = etree.tostring(str_et, method="c14n").decode("utf-8")
    assert normalised_xml_str == page_extractor.get_xml_str()


def test_write_alto_xml_texts():
    alto_extractor = XMLTextlines.from_alto(alto_xml_str)
    for i in range(len(alto_extractor)):
        alto_extractor[i] = alto_extractor[i] + " TEST"
    assert all([txt == alto_str_textlines[i] + " TEST" for i, txt in enumerate(alto_extractor)])
    for i in range(len(alto_extractor)):
        alto_extractor[i] = alto_extractor[i][:-5]
    orig_alto_extractor = XMLTextlines.from_alto(alto_xml_str)
    assert all([txt == orig_alto_extractor[i] for i, txt in enumerate(alto_extractor)])
    for i in range(len(alto_extractor)):
        alto_extractor[i] = ""
    assert all([txt == "" for i, txt in enumerate(alto_extractor)])


def test_write_page_xml_texts():
    page_extractor = XMLTextlines.from_pagexml(page_xml_str)
    for i in range(len(page_extractor)):
        page_extractor[i] = page_extractor[i] + " TEST"
    assert all([txt == page_xml_str_textlines[i] + " TEST" for i, txt in enumerate(page_extractor)])
    for i in range(len(page_extractor)):
        page_extractor[i] = page_extractor[i][:-5]
    orig_page_extractor = XMLTextlines.from_pagexml(page_xml_str)
    assert all([txt == orig_page_extractor[i] for i, txt in enumerate(page_extractor)])
    for i in range(len(page_extractor)):
        page_extractor[i] = ""
    assert all([txt == "" for i, txt in enumerate(page_extractor)])
        

alto_xml_str = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xmlns:tei="http://www.tei-c.org/ns/1.0"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# ../schema/alto.xsd">
   <Description>
      <MeasurementUnit>pixel</MeasurementUnit>
      <sourceImageInformation>
         <fileName>FRAD021_15_H_163_0012.tif</fileName>
      </sourceImageInformation>
   </Description>
   <Layout>
      <Page ID="zone-pb_Fontenay_7" PHYSICAL_IMG_NR="0" WIDTH="4396" HEIGHT="3034">
         <PrintSpace HPOS="0" VPOS="0" WIDTH="4396" HEIGHT="3034">
            <TextBlock HPOS="416" VPOS="343" WIDTH="3645" HEIGHT="2225" ID="zone-col_pb_Fontenay_7_1">
               <Shape>
                  <Polygon POINTS="416,343 4061,343 4061,2568 416,2568"/>
               </Shape>
               <TextLine ID="zone-line_pb_Fontenay_7_1_1" BASELINE="" HPOS="441" VPOS="353" WIDTH="3449" HEIGHT="186">
                  <Shape>
                     <Polygon POINTS="441,353 3890,353 3890,539 441,539"/>
                  </Shape>
                  <String CONTENT="ego steph̃s ,regniaci abbas ,⁊ ego armannus ,eduensis archidiacon9 ,notũ facim9 memorie posteroꝝ qd̃" HPOS="441" VPOS="353" WIDTH="3449" HEIGHT="186"/>
               </TextLine>
               <TextLine ID="zone-line_pb_Fontenay_7_1_2" BASELINE="" HPOS="444" VPOS="546" WIDTH="3510" HEIGHT="168">
                  <Shape>
                     <Polygon POINTS="444,546 3954,546 3954,714 444,714"/>
                  </Shape>
                  <String CONTENT="in p̃sentia nr̃a t̃minata ẽ querela que erat int̃ obertũ de tylio ⁊ fontenet̃ eccłam .movebat ob̃tus calũpniã dicens" HPOS="444" VPOS="546" WIDTH="3510" HEIGHT="168"/>
               </TextLine>
               <TextLine ID="zone-line_pb_Fontenay_7_1_3" BASELINE="" HPOS="441" VPOS="701" WIDTH="3545" HEIGHT="169">
                  <Shape>
                     <Polygon POINTS="441,701 3986,701 3986,870 441,870"/>
                  </Shape>
                  <String CONTENT="t̃ras qͣs monachi habebant a wiłłmo de fraxino debere eẽ de asam̃to suo ,qd̃ monachi econtͣ negabant .nos vͦ apd̃ juna" HPOS="441" VPOS="701" WIDTH="3545" HEIGHT="169"/>
               </TextLine>
            </TextBlock>
         </PrintSpace>
      </Page>
   </Layout>
</alto>
"""

alto_str_textlines = [
    "ego steph̃s ,regniaci abbas ,⁊ ego armannus ,eduensis archidiacon9 ,notũ facim9 memorie posteroꝝ qd̃",
    "in p̃sentia nr̃a t̃minata ẽ querela que erat int̃ obertũ de tylio ⁊ fontenet̃ eccłam .movebat ob̃tus calũpniã dicens",
    "t̃ras qͣs monachi habebant a wiłłmo de fraxino debere eẽ de asam̃to suo ,qd̃ monachi econtͣ negabant .nos vͦ apd̃ juna"]


page_xml_str = """<?xml version="1.0" encoding="UTF-8"  standalone="yes"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd">
  <Metadata>
	<Creator>escriptorium</Creator>
	<Created>2025-09-17T16:44:59.149344+00:00</Created>
        <LastChange>2025-09-17T16:44:59.149364+00:00</LastChange>
  </Metadata>
  <Page imageFilename="BSNSP_12520727_3AAIII_21_r.jpg" imageWidth="2592" imageHeight="3157">
    <TextRegion id="eSc_textblock_d079da01"  custom="structure {type:OldText_eschatocol;}">
      <Coords points="322,2125 287,2868"/>
      <TextLine id="eSc_line_51227ced" >
        <Coords points="502,2230 499,2249 531,2265"/>
        <Baseline points="504,2232 1434,2266"/>
        <TextEquiv><Unicode>Signũ crucis ꝓͥe Guiłłi leõis iudicis ꝗ supͣ</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="eSc_line_834016d0" >
        <Coords points="465,2488 463,2504 584,2504"/>
        <Baseline points="467,2491 1490,2551"/>
        <TextEquiv><Unicode>Ego Leonardus diaconꝰ ⁊ canonic̃ eccłe Mont̃ corbiñ int̃fui ⁊ test̃ sum</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
    <TextRegion id="eSc_textblock_9f977d50"  custom="structure {type:Ornament_invocatio;}">
      <Coords points="367,122 367,316 525,316 525,122"/>
    </TextRegion>
    <TextRegion id="eSc_dummyblock_">
      <TextLine id="eSc_line_e87cc74a" custom="structure {type:default;}">
        <Coords points="365,482 363,501 400,513"/>
        <TextEquiv><Unicode>⁊ uxor pħi đ pagano đ montꝭ corbino. fateor me hr̃. quandam pecciam</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="eSc_line_3d5077e1" custom="structure {type:default;}">
        <Coords points="347,1587 345,1611 422,1610 435,1617"/>
        <TextEquiv><Unicode>am̃ ⁊ si uͦ potuerim̃ ⁊ noluerim̃ ⁊ causari adq̃ aliqͥd exinđ remouer̃ p̃sumserim̃</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="eSc_line_1b85f706" custom="structure {type:default;}">
        <Coords points="333,1660 331,1676 652,1694 654,1694"/>
        <TextEquiv><Unicode>uł hoc breue falsũ ẽe discerim̃. causacio ip̃a sit ꝓrsꝰ tacita ⁊ Inanis ⁊ dupłũ p̃ciũ</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""

page_xml_str_textlines = [
    "Signũ crucis ꝓͥe Guiłłi leõis iudicis ꝗ supͣ",
    "Ego Leonardus diaconꝰ ⁊ canonic̃ eccłe Mont̃ corbiñ int̃fui ⁊ test̃ sum",
    "⁊ uxor pħi đ pagano đ montꝭ corbino. fateor me hr̃. quandam pecciam",
    "am̃ ⁊ si uͦ potuerim̃ ⁊ noluerim̃ ⁊ causari adq̃ aliqͥd exinđ remouer̃ p̃sumserim̃",
    "uł hoc breue falsũ ẽe discerim̃. causacio ip̃a sit ꝓrsꝰ tacita ⁊ Inanis ⁊ dupłũ p̃ciũ"
]