"""人脸检测"""
import base64
from io import BytesIO
from bson import ObjectId
from PIL import Image

from jindai import Plugin
from jindai.models import MediaItem, Paragraph, F, Fn, Var, DbObjectCollection
from plugins.imageproc import MediaItemStage
from plugins.hashing import single_item
from plugins.hashing import bitcount, to_int, whash
from . import facedetectcnn


class FaceDet(MediaItemStage):
    """人脸检测"""

    def crop_faces(self, buf):
        """Crop faces from image in buffer"""
        image = Image.open(buf)
        image.thumbnail((1024, 1024))
        for pos_x, pos_y, width, height, confi in facedetectcnn.facedetect_cnn(image):
            if confi < 75:
                continue
            bufi = image.crop((pos_x, pos_y, pos_x + width, pos_y + height))
            yield bufi

    def resolve_image(self, i: MediaItem, _):
        """Resolve image"""
        data = i.data
        if not data:
            return
        i.faces = []
        for face in self.crop_faces(data):
            i.faces.append(whash(face))

        i.save()
        return i


class FaceDetPlugin(Plugin):
    """人脸检测插件"""

    def __init__(self, app, **_):
        super().__init__(app)
        self.det = FaceDet()
        self.register_pipelines([FaceDet])
        MediaItem.set_field('faces', DbObjectCollection(bytes))

    def handle_page(self, datasource_impl, iid='', fid=''):
        """Handle page"""
        archive = datasource_impl.groups != 'none'

        offset = datasource_impl.skip
        limit = datasource_impl.limit
        datasource_impl.limit = 0
        datasource_impl.raw = False

        if iid == '':
            datasource_impl.aggregator.addFields(
                images=Fn.filter(input=Var.images, as_='item',
                                 cond=Fn.size(Fn.ifNull('$$item.faces', [])))
            ).match(F.images != [])

            result_set = datasource_impl.fetch()
            return result_set

        else:
            fid = 0 if not fid else int(fid)
            iid = ObjectId(iid)

            fdh = [to_int(f) for f in MediaItem.first(F.id == iid).faces]
            if fid:
                fdh = [fdh[fid-1]]
            if not fdh:
                return []

            groupped = {}
            results = []
            for paragraph in datasource_impl.fetch():
                for image_item in paragraph.images:
                    if not image_item or not isinstance(image_item, MediaItem) \
                       or image_item.flag != 0 or not image_item.faces or image_item.id == iid:
                        continue
                    image_item.score = min([
                        min([bitcount(to_int(i) ^ j) for j in fdh])
                        for i in image_item.faces
                    ])
                    rpo = Paragraph(**paragraph.as_dict())
                    rpo.images = [image_item]
                    if archive:
                        groups = [
                            g for g in paragraph.keywords if g.startswith('*')]
                        for group in groups or [paragraph.source['url']]:
                            if group not in groupped or groupped[group][0] > image_item.score:
                                groupped[group] = (image_item.score, rpo)
                    else:
                        results.append((image_item.score, rpo))

            if archive:
                results = list(groupped.values())

            results = [r for _, r in sorted(results, key=lambda x: x[0])[
                offset:offset+limit]]

            paragraph_faces = single_item('', iid)
            source_paragraph = paragraph_faces[0]
            for face in self.det.crop_faces(source_paragraph.images[0].data):
                saved = BytesIO()
                face.save(saved, format='JPEG')
                paragraph_faces.append(
                    Paragraph(
                        _id=source_paragraph.id,
                        images=[
                            MediaItem(source={
                                      'url': 'data:image/jpeg;base64,'
                                      + base64.b64encode(saved.getvalue()).decode('ascii')}, item_type='image')
                        ]
                    )
                )

            if fid:
                paragraph_faces = [paragraph_faces[0], paragraph_faces[fid]]

            return paragraph_faces + [{'spacer': 'spacer'}] + results

    def get_filters(self):
        return {
            'face': {
                'format': 'face/{mediaitem._id}',
                'shortcut': 'e',
                'icon': 'mdi-emoticon-outline'
            }
        }
